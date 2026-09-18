// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>
#include <unordered_map>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <small_gicp/ann/traits.hpp>
#include <small_gicp/ann/knn_result.hpp>
#include <small_gicp/ann/flat_container.hpp>
#include <small_gicp/points/traits.hpp>
#include <small_gicp/util/fast_floor.hpp>
#include <small_gicp/util/vector3i_hash.hpp>

namespace small_gicp {

/// @brief Voxel meta information.
struct VoxelInfo {
public:
  /// @brief Constructor.
  /// @param coord Integer voxel coordinates
  /// @param lru   LRU counter for caching
  VoxelInfo(const Eigen::Vector3i& coord, size_t lru) : lru(lru), coord(coord) {}

public:
  size_t lru;             ///< Last used time
  Eigen::Vector3i coord;  ///< Voxel coordinate
};

/// @brief Incremental voxelmap.
///        This class supports incremental point cloud insertion and LRU-based voxel deletion that removes voxels that are not recently referenced.
/// @note  This class can be used as a point cloud as well as a neighbor search structure.
/// @note  This class can handle arbitrary number of voxels and arbitrary range of voxel coordinates (in 32-bit int range).
/// @note  To use this as a source point cloud for registration, use `SequentialVoxelMapAccessor`.
template <typename VoxelContents>
struct IncrementalVoxelMap {
public:
  using Ptr = std::shared_ptr<IncrementalVoxelMap>;
  using ConstPtr = std::shared_ptr<const IncrementalVoxelMap>;

  /// @brief Constructor.
  /// @param leaf_size  Voxel size
  explicit IncrementalVoxelMap(double leaf_size)
  : inv_leaf_size(1.0 / leaf_size),
    lru_horizon(100),
    lru_clear_cycle(1),
    lru_counter(0),
    sweep_cursor(0),
    eviction_radius(0.0),
    eviction_center(Eigen::Vector3d::Zero()) {
    set_search_offsets(1);
  }

  /// @brief Number of points in the voxelmap.
  size_t size() const { return flat_voxels.size(); }

  /// @brief Insert points to the voxelmap.
  /// @param points Point cloud
  /// @param T      Transformation matrix
  template <typename PointCloud>
  void insert(const PointCloud& points, const Eigen::Isometry3d& T = Eigen::Isometry3d::Identity()) {
    touched_voxels.clear();  // a member, so the insert path allocates nothing

    // Insert points to the voxelmap
    for (size_t i = 0; i < traits::size(points); i++) {
      const Eigen::Vector4d pt = T * traits::point(points, i);
      const Eigen::Vector3i coord = fast_floor(pt * inv_leaf_size).template head<3>();

      auto found = voxels.find(coord);
      if (found == voxels.end()) {
        auto voxel = std::make_shared<std::pair<VoxelInfo, VoxelContents>>(VoxelInfo(coord, lru_counter), VoxelContents());

        // Reuse a tombstoned slot when one is free, so the vector does not
        // grow without bound and live indices never move.
        size_t slot;
        if (!free_slots.empty()) {
          slot = free_slots.back();
          free_slots.pop_back();
          flat_voxels[slot] = voxel;
        } else {
          slot = flat_voxels.size();
          flat_voxels.emplace_back(voxel);
        }
        found = voxels.emplace_hint(found, coord, slot);
      }

      auto& [info, voxel] = *flat_voxels[found->second];
      info.lru = lru_counter;
      // The centre is what a centre-preference policy compares against; the
      // first-come policy ignores it.
      const Eigen::Vector4d cell_center =
          (coord.cast<double>().array() + 0.5).matrix().homogeneous() / inv_leaf_size;
      voxel.add(voxel_setting, pt, points, i, T, cell_center);
      touched_voxels.emplace_back(found->second);
    }

    // Only the voxels this call touched need finalizing; walking the whole map
    // made insertion cost scale with map size instead of with points inserted.
    std::sort(touched_voxels.begin(), touched_voxels.end());
    touched_voxels.erase(std::unique(touched_voxels.begin(), touched_voxels.end()), touched_voxels.end());
    for (const size_t slot : touched_voxels) {
      flat_voxels[slot]->second.finalize();
    }

    ++lru_counter;

    if (lru_clear_cycle <= 1) {
      // Sweep a slice of the map per insert: over lru_horizon inserts the whole
      // map gets checked, no single insert pays O(map), and eviction is spread
      // evenly instead of dropping a tenth of the map at once.
      if (!flat_voxels.empty()) {
        // The sweep rate is tied to the horizon, but a radius rule has no
        // horizon of its own: with a very long lru_horizon the slice would
        // shrink to a few voxels and a voxel could sit outside the radius for
        // thousands of scans. Cap the pass length when the radius is on.
        const size_t pass = (eviction_radius > 0.0) ? std::min<size_t>(lru_horizon, 100) : lru_horizon;
        const size_t horizon = std::max<size_t>(1, pass);
        // 16x oversweep. The cursor would otherwise reach a stale voxel up to a
        // full horizon late, stretching the effective horizon and the map with
        // it; at 16x the lag is horizon/16, which matches what the bulk sweep
        // left behind, so the two eviction policies hold the same window. The
        // extra work is ~18k pointer checks against a ~5 ms insert.
        const size_t slice = std::max<size_t>(1, (16 * flat_voxels.size() + horizon - 1) / horizon);
        for (size_t n = 0; n < slice; n++) {
          sweep_cursor = (sweep_cursor + 1) % flat_voxels.size();
          auto& voxel = flat_voxels[sweep_cursor];
          if (voxel && should_evict(voxel->first)) {
            voxels.erase(voxel->first.coord);
            voxel.reset();
            free_slots.emplace_back(sweep_cursor);
          }
        }
      }
    } else if (lru_counter % lru_clear_cycle == 0) {
      // As vendored: bulk sweep plus a full rehash. Kept so the artefact this
      // causes can be reproduced on purpose.
      auto remove_counter = std::remove_if(flat_voxels.begin(), flat_voxels.end(), [&](const std::shared_ptr<std::pair<VoxelInfo, VoxelContents>>& voxel) {
        return !voxel || should_evict(voxel->first);
      });
      flat_voxels.erase(remove_counter, flat_voxels.end());
      free_slots.clear();

      voxels.clear();
      for (size_t i = 0; i < flat_voxels.size(); i++) {
        voxels[flat_voxels[i]->first.coord] = i;
      }
    }
  }

  /// @brief Whether a voxel should be dropped: stale, or outside the radius.
  bool should_evict(const VoxelInfo& info) const {
    if (info.lru + lru_horizon < lru_counter) {
      return true;
    }
    if (eviction_radius > 0.0) {
      const Eigen::Vector3d center =
          (info.coord.cast<double>().array() + 0.5).matrix() / inv_leaf_size;
      if ((center - eviction_center).squaredNorm() > eviction_radius * eviction_radius) {
        return true;
      }
    }
    return false;
  }

  /// @brief Where the eviction radius is measured from.
  void set_eviction_center(const Eigen::Vector3d& center) { eviction_center = center; }

  /// @brief Find the nearest neighbor.
  /// @param pt       Query point
  /// @param index    Index of the nearest neighbor
  /// @param sq_dist  Squared distance to the nearest neighbor
  /// @return         Number of found points (0 or 1)
  size_t nearest_neighbor_search(const Eigen::Vector4d& pt, size_t* index, double* sq_dist) const {
    const Eigen::Vector3i center = fast_floor(pt * inv_leaf_size).template head<3>();
    size_t voxel_index = 0;
    const auto index_transform = [&](size_t i) { return calc_index(voxel_index, i); };
    KnnResult<1, decltype(index_transform)> result(index, sq_dist, -1, index_transform);

    for (const auto& offset : search_offsets) {
      const Eigen::Vector3i coord = center + offset;
      const auto found = voxels.find(coord);
      if (found == voxels.end()) {
        continue;
      }

      voxel_index = found->second;
      const auto& voxel = flat_voxels[voxel_index]->second;

      traits::Traits<VoxelContents>::knn_search(voxel, pt, result);
    }

    return result.num_found();
  }

  /// @brief Find k nearest neighbors
  /// @param pt          Query point
  /// @param k           Number of neighbors
  /// @param k_indices   Indices of nearest neighbors
  /// @param k_sq_dists  Squared distances to nearest neighbors (sorted in ascending order)
  /// @return            Number of found points
  size_t knn_search(const Eigen::Vector4d& pt, size_t k, size_t* k_indices, double* k_sq_dists) const {
    const Eigen::Vector3i center = fast_floor(pt * inv_leaf_size).template head<3>();

    size_t voxel_index = 0;
    const auto index_transform = [&](size_t i) { return calc_index(voxel_index, i); };
    KnnResult<-1, decltype(index_transform)> result(k_indices, k_sq_dists, k, index_transform);

    for (const auto& offset : search_offsets) {
      const Eigen::Vector3i coord = center + offset;
      const auto found = voxels.find(coord);
      if (found == voxels.end()) {
        continue;
      }

      voxel_index = found->second;
      const auto& voxel = flat_voxels[voxel_index]->second;

      traits::Traits<VoxelContents>::knn_search(voxel, pt, result);
    }

    return result.num_found();
  }

  /// @brief Calculate the global point index from the voxel index and the point index.
  inline size_t calc_index(const size_t voxel_id, const size_t point_id) const { return (voxel_id << point_id_bits) | point_id; }
  inline size_t voxel_id(const size_t i) const { return i >> point_id_bits; }                 ///< Extract the point ID from a global index.
  inline size_t point_id(const size_t i) const { return i & ((1ull << point_id_bits) - 1); }  ///< Extract the voxel ID from a global index.

  /// @brief Set the pattern of the search offsets. (Must be 1, 7, or 27)
  /// @note  1: center only, 7: center + 6 adjacent neighbors (+- 1X/1Y/1Z), 27: center + 26 neighbors (3 x 3 x 3 cube)
  void set_search_offsets(int num_offsets) {
    switch (num_offsets) {
      default:
        std::cerr << "warning: unsupported search_offsets=" << num_offsets << " (supported values: 1, 7, 27)" << std::endl;
        std::cerr << "       : using default search_offsets=1" << std::endl;
        [[fallthrough]];
      case 1:
        search_offsets = {Eigen::Vector3i(0, 0, 0)};
        break;
      case 7:
        search_offsets = {
          Eigen::Vector3i(0, 0, 0),
          Eigen::Vector3i(1, 0, 0),
          Eigen::Vector3i(0, 1, 0),
          Eigen::Vector3i(0, 0, 1),
          Eigen::Vector3i(-1, 0, 0),
          Eigen::Vector3i(0, -1, 0),
          Eigen::Vector3i(0, 0, -1)};
        break;
      case 27:
        for (int i = -1; i <= 1; i++) {
          for (int j = -1; j <= 1; j++) {
            for (int k = -1; k <= 1; k++) {
              search_offsets.emplace_back(i, j, k);
            }
          }
        }
        break;
    }
  }

public:
  static_assert(sizeof(size_t) == 8, "size_t must be 64-bit");
  static constexpr int point_id_bits = 32;                  ///< Use the first 32 bits for point id
  static constexpr int voxel_id_bits = 64 - point_id_bits;  ///< Use the remaining bits for voxel id
  const double inv_leaf_size;                               ///< Inverse of the voxel size

  size_t lru_horizon;      ///< LRU horizon size. Voxels that have not been accessed for lru_horizon steps are deleted.
  size_t lru_clear_cycle;  ///< LRU clear cycle. Voxel deletion is performed every lru_clear_cycle steps.
  size_t lru_counter;      ///< LRU counter. Incremented every step.
  size_t sweep_cursor;     ///< Rolling position of the incremental eviction sweep.

  double eviction_radius;          ///< Drop voxels beyond this of eviction_center. 0 disables it.
  Eigen::Vector3d eviction_center;  ///< Where that radius is measured from, updated per scan.

  std::vector<Eigen::Vector3i> search_offsets;  ///< Voxel search offsets.

  typename VoxelContents::Setting voxel_setting;                                  ///< Voxel setting.
  std::vector<std::shared_ptr<std::pair<VoxelInfo, VoxelContents>>> flat_voxels;  ///< Voxel contents.
  std::unordered_map<Eigen::Vector3i, size_t, XORVector3iHash> voxels;            ///< Voxel index map.
  std::vector<size_t> free_slots;                                                 ///< Tombstoned slots, reusable by new voxels.
  std::vector<size_t> touched_voxels;                                             ///< Scratch: slots touched by the last insert.
};

namespace traits {

template <typename VoxelContents>
struct Traits<IncrementalVoxelMap<VoxelContents>> {
  static size_t size(const IncrementalVoxelMap<VoxelContents>& voxelmap) { return voxelmap.size(); }

  static Eigen::Vector4d point(const IncrementalVoxelMap<VoxelContents>& voxelmap, size_t i) {
    const size_t voxel_id = voxelmap.voxel_id(i);
    const size_t point_id = voxelmap.point_id(i);
    const auto& voxel = voxelmap.flat_voxels[voxel_id]->second;
    return traits::point(voxel, point_id);
  }
  static Eigen::Vector4d normal(const IncrementalVoxelMap<VoxelContents>& voxelmap, size_t i) {
    const size_t voxel_id = voxelmap.voxel_id(i);
    const size_t point_id = voxelmap.point_id(i);
    const auto& voxel = voxelmap.flat_voxels[voxel_id]->second;
    return traits::normal(voxel, point_id);
  }
  static Eigen::Matrix4d cov(const IncrementalVoxelMap<VoxelContents>& voxelmap, size_t i) {
    const size_t voxel_id = voxelmap.voxel_id(i);
    const size_t point_id = voxelmap.point_id(i);
    const auto& voxel = voxelmap.flat_voxels[voxel_id]->second;
    return traits::cov(voxel, point_id);
  }

  static size_t nearest_neighbor_search(const IncrementalVoxelMap<VoxelContents>& voxelmap, const Eigen::Vector4d& pt, size_t* k_index, double* k_sq_dist) {
    return voxelmap.nearest_neighbor_search(pt, k_index, k_sq_dist);
  }

  static size_t knn_search(const IncrementalVoxelMap<VoxelContents>& voxelmap, const Eigen::Vector4d& pt, int k, size_t* k_index, double* k_sq_dist) {
    return voxelmap.knn_search(pt, k, k_index, k_sq_dist);
  }
};

template <typename VoxelContents>
std::vector<size_t> point_indices(const IncrementalVoxelMap<VoxelContents>& voxelmap) {
  std::vector<size_t> indices;
  indices.reserve(voxelmap.size() * 5);

  for (size_t voxel_id = 0; voxel_id < voxelmap.flat_voxels.size(); voxel_id++) {
    const auto& voxel = voxelmap.flat_voxels[voxel_id];
    if (!voxel) {
      continue;  // tombstoned by an eviction
    }
    for (size_t point_id = 0; point_id < traits::size(voxel->second); point_id++) {
      indices.emplace_back(voxelmap.calc_index(voxel_id, point_id));
    }
  }

  return indices;
}

template <typename VoxelContents>
std::vector<Eigen::Vector4d> voxel_points(const IncrementalVoxelMap<VoxelContents>& voxelmap) {
  std::vector<Eigen::Vector4d> points;
  points.reserve(voxelmap.size() * 5);

  for (const auto& voxel : voxelmap.flat_voxels) {
    if (!voxel) {
      continue;  // tombstoned by an eviction
    }
    for (size_t i = 0; i < traits::size(voxel->second); i++) {
      points.push_back(traits::point(voxel->second, i));
    }
  }
  return points;
}

template <typename VoxelContents>
std::vector<Eigen::Vector4d> voxel_normals(const IncrementalVoxelMap<VoxelContents>& voxelmap) {
  std::vector<Eigen::Vector4d> normals;
  normals.reserve(voxelmap.size() * 5);

  for (const auto& voxel : voxelmap.flat_voxels) {
    if (!voxel) {
      continue;  // tombstoned by an eviction
    }
    for (size_t i = 0; i < traits::size(voxel->second); i++) {
      normals.push_back(traits::normal(voxel->second, i));
    }
  }
  return normals;
}

template <typename VoxelContents>
std::vector<Eigen::Matrix4d> voxel_covs(const IncrementalVoxelMap<VoxelContents>& voxelmap) {
  std::vector<Eigen::Matrix4d> covs;
  covs.reserve(voxelmap.size() * 5);

  for (const auto& voxel : voxelmap.flat_voxels) {
    if (!voxel) {
      continue;  // tombstoned by an eviction
    }
    for (size_t i = 0; i < traits::size(voxel->second); i++) {
      covs.push_back(traits::cov(voxel->second, i));
    }
  }
  return covs;
}

}  // namespace traits

}  // namespace small_gicp
