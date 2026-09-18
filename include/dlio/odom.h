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

#include "dlio/dlio.h"
#include "dlio/ivox_target.h"

#include <jsk_rviz_plugins/OverlayText.h>

class dlio::OdomNode {

public:

  OdomNode(ros::NodeHandle node_handle);
  ~OdomNode();

  void start();

private:

  struct State;
  struct ImuMeas;

  void getParams();

  void callbackPointCloud(const sensor_msgs::PointCloud2ConstPtr& pc);
  void callbackImu(const sensor_msgs::Imu::ConstPtr& imu);

  void publishPose(const ros::TimerEvent& e);

  void publishToROS(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud);
  void publishCloud(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud);
  void publishKeyframe(std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>,
                       pcl::PointCloud<PointType>::ConstPtr> kf, ros::Time timestamp);

  void getScanFromROS(const sensor_msgs::PointCloud2ConstPtr& pc);
  void preprocessPoints();
  void deskewPointcloud();
  void initializeInputTarget();
  void setInputSource();

  void initializeDLIO();

  void getNextPose();
  bool imuMeasFromTimeRange(double start_time, double end_time,
                            boost::circular_buffer<ImuMeas>::reverse_iterator& begin_imu_it,
                            boost::circular_buffer<ImuMeas>::reverse_iterator& end_imu_it);
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
    integrateImu(double start_time, Eigen::Quaternionf q_init, Eigen::Vector3f p_init, Eigen::Vector3f v_init,
                 const std::vector<double>& sorted_timestamps);
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
    integrateImuInternal(Eigen::Quaternionf q_init, Eigen::Vector3f p_init, Eigen::Vector3f v_init,
                         const std::vector<double>& sorted_timestamps,
                         boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it,
                         boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it);
  void propagateGICP();

  void propagateState();
  void updateState();

  void setAdaptiveParams();
  void setKeyframeCloud();

  void computeMetrics();
  void computeSpaciousness();
  void computeDensity();

  sensor_msgs::Imu::Ptr transformImu(const sensor_msgs::Imu::ConstPtr& imu);

  void updateKeyframes();
  void computeConvexHull();
  void computeConcaveHull();
  void pushSubmapIndices(std::vector<float> dists, int k, std::vector<int> frames);
  void buildSubmap(State vehicle_state);
  void insertScanIntoIVox();
  void updateAdaptiveLeaf(const pcl::PointCloud<PointType>& probe);
  double correspondenceFromLeaf() const;
  void publishIVox();
  void publishStats();
  void buildKeyframesAndSubmap(State vehicle_state);
  void pauseSubmapBuildIfNeeded();

  void debug();

  ros::NodeHandle nh;
  ros::Timer publish_timer;

  // Subscribers
  ros::Subscriber lidar_sub;
  ros::Subscriber imu_sub;

  // Publishers
  ros::Publisher odom_pub;
  ros::Publisher pose_pub;
  ros::Publisher path_pub;
  ros::Publisher kf_pose_pub;
  ros::Publisher kf_cloud_pub;
  ros::Publisher deskewed_pub;
  // The submap GICP registers against, and which keyframes it was built from.
  // Published whenever the selection changes, so a wrong pick (a keyframe from
  // the floor below, say) shows up as the cloud and the markers it came from.
  ros::Publisher submap_pub;
  ros::Publisher submap_kf_pub;
  ros::Publisher stats_pub;

  // ROS Msgs
  nav_msgs::Odometry odom_ros;
  geometry_msgs::PoseStamped pose_ros;
  nav_msgs::Path path_ros;
  geometry_msgs::PoseArray kf_pose_ros;

  // Flags
  std::atomic<bool> dlio_initialized;
  std::atomic<bool> first_valid_scan;
  std::atomic<bool> first_imu_received;
  std::atomic<bool> imu_calibrated;
  std::atomic<bool> submap_hasChanged;
  std::atomic<bool> gicp_hasConverged;
  std::atomic<bool> deskew_status;
  std::atomic<int> deskew_size;

  // Threads
  std::thread publish_thread;
  std::thread publish_keyframe_thread;
  std::thread metrics_thread;
  std::thread debug_thread;

  // Trajectory
  std::vector<std::pair<Eigen::Vector3f, Eigen::Quaternionf>> trajectory;

  // Estimated trajectory written as it runs, in the same TUM layout the
  // SuperLoc ground truth uses: stamp tx ty tz qx qy qz qw.
  std::string trajectory_path_;
  std::ofstream trajectory_file;
  void writeTrajectoryPose();
  double length_traversed;

  // Keyframes
  std::vector<std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>,
                        pcl::PointCloud<PointType>::ConstPtr>> keyframes;
  std::vector<ros::Time> keyframe_timestamps;
  std::vector<std::shared_ptr<const nano_gicp::CovarianceList>> keyframe_normals;
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> keyframe_transformations;
  std::mutex keyframes_mutex;

  // Sensor Type
  dlio::SensorType sensor;

  // Frames
  std::string odom_frame;
  std::string baselink_frame;
  std::string lidar_frame;
  std::string imu_frame;

  // Preprocessing
  pcl::CropBox<PointType> crop;
  pcl::VoxelGrid<PointType> voxel;

  // Point Clouds
  pcl::PointCloud<PointType>::ConstPtr original_scan;
  pcl::PointCloud<PointType>::ConstPtr deskewed_scan;
  pcl::PointCloud<PointType>::ConstPtr current_scan;

  // Keyframes
  pcl::PointCloud<PointType>::ConstPtr keyframe_cloud;
  int num_processed_keyframes;

  pcl::ConvexHull<PointType> convex_hull;
  pcl::ConcaveHull<PointType> concave_hull;
  std::vector<int> keyframe_convex;
  std::vector<int> keyframe_concave;

  // Submap
  pcl::PointCloud<PointType>::ConstPtr submap_cloud;
  std::shared_ptr<const nano_gicp::CovarianceList> submap_normals;
  std::shared_ptr<const nanoflann::KdTreeFLANN<PointType>> submap_kdtree;

  std::vector<int> submap_kf_idx_curr;
  std::vector<int> submap_kf_idx_prev;

  // GICP target map backend: "submap" (keyframes + kd-tree) or "ivox"
  // (an incremental voxel map updated in place after every scan).
  std::string map_backend_;
  std::shared_ptr<dlio::IVoxMap> ivox_map;
  double ivox_resolution_;
  double ivox_min_dist_in_cell_;
  int ivox_max_points_in_cell_;
  int ivox_lru_horizon_;
  int ivox_lru_clear_cycle_;
  std::string ivox_insert_policy_;
  int ivox_search_offsets_;
  double ivox_insert_downsample_;
  double ivox_eviction_radius_;

  // Keep the voxelized scan inside a point-count band. The fixed leaf is used
  // first; too many points are sampled down, too few are re-voxelized finer.
  bool   scan_points_enabled_;
  int    scan_points_min_;
  int    scan_points_max_;
  double scan_points_fine_leaf_;
  int    scan_points_used_;

  // Scale-aware adaptive voxelization, after GenZ-LIO (arXiv:2603.16273).
  // A PD controller drives the voxel leaf so the voxelized scan holds a point
  // count that suits how open the scene is.
  bool   target_points_enabled_;
  int    target_points_min_;
  int    target_points_max_;
  double target_open_range_;
  double target_exponent_;
  int    target_window_;
  double target_leaf_min_;
  double target_leaf_max_;
  double target_lambda_p_;
  double target_lambda_d_;
  double target_kp_min_;
  double target_kp_max_;
  double target_kd_min_;
  double target_kd_max_;
  double adaptive_leaf_;
  double last_target_points_;
  std::deque<double> range_window_;
  double target_prev_error_;
  bool   target_have_prev_error_;

  // Tie the GICP correspondence distance to the leaf the scan was voxelized
  // with, instead of the two-valued spaciousness step.
  bool   corr_from_leaf_;
  double corr_leaf_ratio_;

  bool new_submap_is_ready;
  std::future<void> submap_future;
  std::condition_variable submap_build_cv;
  bool main_loop_running;
  std::mutex main_loop_running_mutex;

  // Timestamps
  ros::Time scan_header_stamp;
  double scan_stamp;
  double prev_scan_stamp;
  double scan_dt;
  std::vector<double> comp_times;
  std::vector<double> imu_rates;
  std::vector<double> lidar_rates;

  double first_scan_stamp;
  double elapsed_time;

  // GICP
  nano_gicp::NanoGICP<PointType, PointType> gicp;
  nano_gicp::NanoGICP<PointType, PointType> gicp_temp;

  // Transformations
  Eigen::Matrix4f T, T_prior, T_corr;
  Eigen::Quaternionf q_final;

  Eigen::Vector3f origin;

  struct Extrinsics {
    struct SE3 {
      Eigen::Vector3f t;
      Eigen::Matrix3f R;
    };
    SE3 baselink2imu;
    SE3 baselink2lidar;
    Eigen::Matrix4f baselink2imu_T;
    Eigen::Matrix4f baselink2lidar_T;
  }; Extrinsics extrinsics;

  // IMU
  ros::Time imu_stamp;
  double first_imu_stamp;
  double prev_imu_stamp;
  double imu_dp, imu_dq_deg;

  struct ImuMeas {
    double stamp;
    double dt; // defined as the difference between the current and the previous measurement
    Eigen::Vector3f ang_vel;
    Eigen::Vector3f lin_accel;
  }; ImuMeas imu_meas;

  boost::circular_buffer<ImuMeas> imu_buffer;
  std::mutex mtx_imu;
  std::condition_variable cv_imu_stamp;

  static bool comparatorImu(ImuMeas m1, ImuMeas m2) {
    return (m1.stamp < m2.stamp);
  };

  // Geometric Observer
  struct Geo {
    bool first_opt_done;
    std::mutex mtx;
    double dp;
    double dq_deg;
    Eigen::Vector3f prev_p;
    Eigen::Quaternionf prev_q;
    Eigen::Vector3f prev_vel;
  }; Geo geo;

  // State Vector
  struct ImuBias {
    Eigen::Vector3f gyro;
    Eigen::Vector3f accel;
  };

  struct Frames {
    Eigen::Vector3f b;
    Eigen::Vector3f w;
  };

  struct Velocity {
    Frames lin;
    Frames ang;
  };

  struct State {
    Eigen::Vector3f p; // position in world frame
    Eigen::Quaternionf q; // orientation in world frame
    Velocity v;
    ImuBias b; // imu biases in body frame
  }; State state;

  struct Pose {
    Eigen::Vector3f p; // position in world frame
    Eigen::Quaternionf q; // orientation in world frame
  };
  Pose lidarPose;
  Pose imuPose;

  // Metrics
  struct Metrics {
    std::vector<float> spaciousness;
    std::vector<float> density;
  }; Metrics metrics;

  std::string cpu_type;
  std::vector<double> cpu_percents;
  clock_t lastCPU, lastSysCPU, lastUserCPU;
  int numProcessors;

  // Parameters
  std::string version_;
  int num_threads_;
  bool verbose;

  bool deskew_;

  double gravity_;

  bool time_offset_;

  bool adaptive_params_;

  double obs_submap_thresh_;
  double obs_keyframe_thresh_;
  double obs_keyframe_lag_;

  double keyframe_thresh_dist_;
  double keyframe_thresh_rot_;

  int submap_knn_;
  int submap_kcv_;
  int submap_kcc_;
  double submap_concave_alpha_;

  bool densemap_filtered_;
  bool wait_until_move_;

  double crop_size_;

  bool vf_use_;
  double vf_res_;

  bool imu_calibrate_;
  bool calibrate_gyro_;
  bool calibrate_accel_;
  bool gravity_align_;
  double imu_calib_time_;
  int imu_buffer_size_;
  Eigen::Matrix3f imu_accel_sm_;

  int gicp_min_num_points_;
  int gicp_k_correspondences_;
  double gicp_max_corr_dist_;
  int gicp_max_iter_;
  double gicp_transformation_ep_;
  double gicp_rotation_ep_;
  double gicp_init_lambda_factor_;

  double geo_Kp_;
  double geo_Kv_;
  double geo_Kq_;
  double geo_Kab_;
  double geo_Kgb_;
  double geo_abias_max_;
  double geo_gbias_max_;

};
