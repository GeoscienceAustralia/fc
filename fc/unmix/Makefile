FC := gfortran
F2PY:= f2py

all: unmiximage.pyf unmiximage.so

unmiximage.pyf unmiximage.so:
	$(F2PY) -h unmiximage.pyf -m unmiximage unmiximage.f90
	$(F2PY) -c -m unmiximage unmiximage.pyf constants_NSWC.f90 nnls.f90 unmiximage.f90

clean:
	@-rm -f *.so
	@-rm -f *.o
	@-rm -f *.mod
	@-rm -f *.pyf
	@-rm -fr build