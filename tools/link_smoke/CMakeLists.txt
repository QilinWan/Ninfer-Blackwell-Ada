# Links the whole engine archive set into one executable without touching a device, so a
# configuration that compiles but cannot link fails here instead of in a downstream project.
add_executable(ninfer-link-smoke main.cpp)
target_link_libraries(ninfer-link-smoke PRIVATE ninfer::engine)
