
CODE_DIR = fc/unmix

all: unmix


unmix:
	$(MAKE) -C $(CODE_DIR)

clean:
	$(MAKE) -C $(CODE_DIR) clean
