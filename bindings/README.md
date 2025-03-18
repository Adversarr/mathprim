# pymp: python bindings for mathprim

# Build

First, fetch and prepare all the cpp source.
```sh
./get-csrc.sh
```

use `poetry` to package the library, and you will find the packaged wheel in `dist` folder, just install the wheel.

```sh
poetry build
pip install dist/pymp-XXXX.whl
```

`XXXX` depends on your platform.