/***********************************************************
 *                                                         *
 * Copyright (c)                                           *
 *                                                         *
 * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
 * University of California, Los Angeles                   *
 *                                                         *
 * Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez   *
 * Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu         *
 *                                                         *
 ***********************************************************/

/***********************************************************************
 * BSD 3-Clause License
 *
 * Copyright (c) 2020, SMRT-AIST
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#pragma once

#include <cstdint>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/registration.h>

#include <small_gicp/ann/flat_container.hpp>
#include <small_gicp/ann/incremental_voxelmap.hpp>

#include "nano_gicp/lsq_registration.h"
#include "nano_gicp/nanoflann_adaptor.h"

namespace nano_gicp {

typedef std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> CovarianceList;

// Alternative target map: an incremental voxel map carrying points and their
// covariances. When one is set, correspondence search goes through it instead
// of the target cloud + kd-tree pair. Mahalanobis weighting, linearization and
// the solver downstream are untouched.
typedef small_gicp::IncrementalVoxelMap<small_gicp::FlatContainer<false, true>> IVoxTarget;

enum class RegularizationMethod { NONE, MIN_EIG, NORMALIZED_MIN_EIG, PLANE, FROBENIUS };

template<typename PointSource, typename PointTarget>
class NanoGICP : public LsqRegistration<PointSource, PointTarget> {
public:
  using Scalar = float;
  using Matrix4 = typename pcl::Registration<PointSource, PointTarget, Scalar>::Matrix4;

  using PointCloudSource = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudSource;
  using PointCloudSourcePtr = typename PointCloudSource::Ptr;
  using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

  using PointCloudTarget = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudTarget;
  using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
  using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

protected:
  using pcl::Registration<PointSource, PointTarget, Scalar>::reg_name_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::input_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::target_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::final_transformation_;

public:
  NanoGICP();
  virtual ~NanoGICP() override;

  void setNumThreads(int n);
  void setCorrespondenceRandomness(int k);
  void setMaxCorrespondenceDistance(double corr);
  void setRegularizationMethod(RegularizationMethod method);

  virtual void swapSourceAndTarget() override;
  virtual void clearSource() override;
  virtual void clearTarget() override;

  virtual void setInputSource(const PointCloudSourceConstPtr& cloud) override;
  virtual void setSourceCovariances(const std::shared_ptr<const CovarianceList>& covs);
  virtual void setInputTarget(const PointCloudTargetConstPtr& cloud) override;
  virtual void setTargetCovariances(const std::shared_ptr<const CovarianceList>& covs);

  virtual void registerInputSource(const PointCloudSourceConstPtr& cloud);
  virtual void registerInputTarget(const PointCloudTargetConstPtr& cloud);

  virtual bool calculateSourceCovariances();
  virtual bool calculateTargetCovariances();

  std::shared_ptr<const CovarianceList> getSourceCovariances() const {
    return source_covs_;
  }

  std::shared_ptr<const CovarianceList> getTargetCovariances() const {
    return target_covs_;
  }

  // This class keeps its own corr_dist_threshold_, which shadows the one in
  // pcl::Registration. The base getter therefore reports pcl's untouched
  // default, not the distance this class actually matches against.
  double maxCorrespondenceDistance() const { return corr_dist_threshold_; }

  // Use an iVox as the target map. Pass nullptr to fall back to the kd-tree.
  void setTargetIVox(const std::shared_ptr<const IVoxTarget>& ivox);

  // Skip the kd-tree pcl::Registration::initCompute() builds over the target.
  // Nothing in this class searches it, so building it is pure overhead.
  void setSkipPclTargetTree(bool skip) { this->force_no_recompute_ = skip; }

  virtual void update_correspondences(const Eigen::Isometry3d& trans);

protected:
  virtual void computeTransformation(PointCloudSource& output, const Matrix4& guess) override;

  virtual double linearize(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) override;

  virtual double compute_error(const Eigen::Isometry3d& trans) override;

  template<typename PointT>
  bool calculate_covariances(const typename pcl::PointCloud<PointT>::ConstPtr& cloud, const nanoflann::KdTreeFLANN<PointT>& kdtree, CovarianceList& covariances, float& density);

public:
  std::shared_ptr<const nanoflann::KdTreeFLANN<PointSource>> source_kdtree_;
  std::shared_ptr<const nanoflann::KdTreeFLANN<PointTarget>> target_kdtree_;

  std::shared_ptr<const CovarianceList> source_covs_;
  std::shared_ptr<const CovarianceList> target_covs_;

  std::shared_ptr<const IVoxTarget> target_ivox_;

  float source_density_;
  float target_density_;

  int num_correspondences;

protected:
  int num_threads_;
  int k_correspondences_;
  double corr_dist_threshold_;

  RegularizationMethod regularization_method_;

  CovarianceList mahalanobis_;

  std::vector<std::int64_t> correspondences_;
  std::vector<float> sq_distances_;
};
}  // namespace nano_gicp

