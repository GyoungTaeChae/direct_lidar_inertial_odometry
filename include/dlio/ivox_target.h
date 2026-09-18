/***********************************************************
 *                                                         *
 * iVox target map for GICP. Wraps small_gicp's             *
 * IncrementalVoxelMap so it can stand in for the keyframe  *
 * submap + nanoflann kd-tree behind the same contract      *
 * NanoGICP needs: nearest-1 lookup, and point/covariance   *
 * by index.                                                *
 *                                                         *
 ***********************************************************/

#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>

#include <small_gicp/ann/flat_container.hpp>
#include <small_gicp/ann/incremental_voxelmap.hpp>

#include "nano_gicp/nano_gicp.h"

namespace dlio {

// A scan paired with the per-point covariances GICP already computed for it as
// the source cloud. Handing both to the map lets the map reuse those instead of
// recomputing, so switching backends does not change what a covariance means.
template <typename PointT>
struct ScanWithCovariances {
  const pcl::PointCloud<PointT>* cloud;
  const nano_gicp::CovarianceList* covariances;
};

// Points + covariances, no normals.
using IVoxContainer = small_gicp::FlatContainer<false, true>;
using IVoxMap = small_gicp::IncrementalVoxelMap<IVoxContainer>;

}  // namespace dlio

namespace small_gicp {
namespace traits {

template <typename PointT>
struct Traits<dlio::ScanWithCovariances<PointT>> {
  using Scan = dlio::ScanWithCovariances<PointT>;

  static size_t size(const Scan& scan) { return scan.cloud->size(); }
  static bool has_points(const Scan& scan) { return !scan.cloud->empty(); }
  static bool has_normals(const Scan&) { return false; }
  static bool has_covs(const Scan& scan) { return scan.covariances != nullptr; }

  static Eigen::Vector4d point(const Scan& scan, size_t i) {
    const auto& p = scan.cloud->points[i];
    return Eigen::Vector4d(p.x, p.y, p.z, 1.0);
  }

  static const Eigen::Matrix4d& cov(const Scan& scan, size_t i) {
    return (*scan.covariances)[i];
  }
};

}  // namespace traits
}  // namespace small_gicp
