cmake_minimum_required(VERSION 3.18)

project(unmiximage LANGUAGES Fortran C)

# Find Python 3 and its development headers
find_package(
    Python3 
    COMPONENTS 
    Development.Module
    NumPy)

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
    OUTPUT_NAME "unmiximage"
)

# Python3::Module supplies the Python extension-module development settings.
# Python3::NumPy supplies the NumPy include directory.
target_link_libraries(
    unmiximage
    PRIVATE
        Python3::Module
        Python3::NumPy
)

# Install the shared object to the Python package directory
install(
    TARGETS unmiximage
    LIBRARY DESTINATION fc/unmix
)
