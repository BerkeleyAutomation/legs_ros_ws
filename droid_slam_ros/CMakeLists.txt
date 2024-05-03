cmake_minimum_required(VERSION 3.8)
project(droid_slam_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/multi_ptcloud_d455_2zed_global_real_droid_subscriber_node_prime.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/space_bundle_adjust_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/convert_grocery_store_to_bag.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/grocery_store_global_real_droid_subscriber_node_prime.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/view_multi_compressed_bag.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/convert_d455_2_zed_folder_to_bag.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  droid_slam
  DESTINATION share/${PROJECT_NAME}
)

install(FILES
  droid.pth
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
