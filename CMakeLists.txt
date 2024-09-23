cmake_minimum_required(VERSION 3.15)

project(unmiximage LANGUAGES Fortran C)

# Find Python 3 and its development headers
find_package(Python3 COMPONENTS Development NumPy REQUIRED)

# Define the source files (both Fortran and C)
set(SOURCES
    ${CMAKE_SOURCE_DIR}/fc/unmix/unmiximage.f90
    ${CMAKE_SOURCE_DIR}/fc/unmix/constants_NSWC.f90
    ${CMAKE_SOURCE_DIR}/fc/unmix/nnls.f90
    ${CMAKE_SOURCE_DIR}/fc/unmix/unmiximage_wrapper.c  # C wrapper
)

# Create the shared library
add_library(unmiximage MODULE ${SOURCES})

set_target_properties(unmiximage PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)

# Link Python headers and libraries to the target
target_include_directories(unmiximage PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(unmiximage PRIVATE ${Python3_LIBRARIES})

# Install the shared object to the Python package directory
install(TARGETS unmiximage
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/fc/unmix)
