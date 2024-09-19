#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdio.h>

// Declare the Fortran subroutine
extern void unmiximage_(
    double *image,
    double *endMemberMatrix,
    int *numBands,
    int *numEndMembers,
    int *numRows,
    int *numCols,
    double *inNull,
    double *outNull,
    double *fractionsImage
);

// Python wrapper function that only takes 4 inputs
static PyObject* py_unmiximage(PyObject* self, PyObject* args) {
    PyArrayObject *image_obj, *endMemberMatrix_obj, *fractionsImage_obj;
    double inNull, outNull;

    if (!PyArg_ParseTuple(args, "O!O!dd",
                          &PyArray_Type, &image_obj,           // Input image
                          &PyArray_Type, &endMemberMatrix_obj, // End member matrix
                          &inNull, &outNull                    // Scalars: inNull, outNull
    )) {
        PyErr_SetString(PyExc_TypeError, "Failed to parse input arguments. Expecting two NumPy arrays followed by two doubles.");
        return NULL;  // Return NULL if parsing fails
    }

    // Ensure the input arrays are of type double (NPY_DOUBLE)
    if (PyArray_TYPE(image_obj) != NPY_DOUBLE || PyArray_TYPE(endMemberMatrix_obj) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "Input arrays must be of type double.");
        return NULL;
    }

    // Ensure the image array is Fortran-contiguous (column-major)
    if (!PyArray_ISFARRAY(image_obj)) {
        image_obj = (PyArrayObject*) PyArray_FromAny(image_obj,
                                                     PyArray_DescrFromType(NPY_DOUBLE),
                                                     3, 3, NPY_ARRAY_F_CONTIGUOUS, NULL);
        if (image_obj == NULL) {
            PyErr_SetString(PyExc_ValueError, "Failed to convert image to Fortran-contiguous.");
            return NULL;
        }
    }

    // Ensure the endMemberMatrix array is Fortran-contiguous (column-major)
    if (!PyArray_ISFARRAY(endMemberMatrix_obj)) {
        endMemberMatrix_obj = (PyArrayObject*) PyArray_FromAny(endMemberMatrix_obj,
                                                               PyArray_DescrFromType(NPY_DOUBLE),
                                                               2, 2, NPY_ARRAY_F_CONTIGUOUS, NULL);
        if (endMemberMatrix_obj == NULL) {
            PyErr_SetString(PyExc_ValueError, "Failed to convert endMemberMatrix to Fortran-contiguous.");
            Py_XDECREF(image_obj);
            return NULL;
        }
    }

    // Check if conversion failed (NULL result)
    if (image_obj == NULL || endMemberMatrix_obj == NULL) {
        PyErr_SetString(PyExc_ValueError, "Failed to convert arrays to Fortran order.");
        Py_XDECREF(image_obj);
        Py_XDECREF(endMemberMatrix_obj);
        return NULL;
    }

    // Get the dimensions of the input arrays
    npy_intp *image_dims = PyArray_DIMS(image_obj);
    npy_intp *endMemberMatrix_dims = PyArray_DIMS(endMemberMatrix_obj);

    //numBands,numRows,numCols
    int numRows = (int)image_dims[1];
    int numCols = (int)image_dims[2];
    int numBands = (int)image_dims[0];
    // endMemberMatrix is (numBands, numEndMembers)
    int numEndMembers = (int)endMemberMatrix_dims[1];  

    // Create the output array (fractionsImage) with the dimensions (numEndMembers+1,numRows,numCols)
    // and fortran-contiguous
    npy_intp fractions_dims[3] = {numEndMembers+1, numRows, numCols};
    fractionsImage_obj = (PyArrayObject*) PyArray_ZEROS(3, fractions_dims, NPY_DOUBLE, 1);

    if (fractionsImage_obj == NULL) {
    	PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for the output array");
        return NULL;  // Error creating the output array
    }

    // Get pointers to the underlying C data in the NumPy arrays
    double *image = (double*) PyArray_DATA(image_obj);
    double *endMemberMatrix = (double*) PyArray_DATA(endMemberMatrix_obj);
    double *fractionsImage = (double*) PyArray_DATA(fractionsImage_obj);


    // Call the Fortran subroutine with the deduced dimensions and input/output arrays
    unmiximage_(image, endMemberMatrix, &numBands, &numEndMembers, &numRows, &numCols, &inNull, &outNull, fractionsImage);

    // Convert array to c-contiguous before return
    if (!PyArray_ISCARRAY(fractionsImage_obj)) {
        fractionsImage_obj = (PyArrayObject*) PyArray_FromArray(fractionsImage_obj,
                                                                  PyArray_DescrFromType(NPY_DOUBLE),
                                                                  NPY_ARRAY_C_CONTIGUOUS);
        if (!fractionsImage_obj) {
            PyErr_SetString(PyExc_ValueError, "Failed to convert fractionsImage to C-contiguous.");
            return NULL;
        }
    }

    return PyArray_Return(fractionsImage_obj);
}

static PyMethodDef UnmixImageMethods[] = {
    {"unmiximage", py_unmiximage, METH_VARARGS, "Perform unmixing of image with end member matrix using Fortran subroutine"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef unmiximagemodule = {
    PyModuleDef_HEAD_INIT,
    "unmiximage",
    NULL,
    -1,
    UnmixImageMethods
};

PyMODINIT_FUNC PyInit_unmiximage(void) {
    import_array();  // Initialize NumPy
    return PyModule_Create(&unmiximagemodule);
}
