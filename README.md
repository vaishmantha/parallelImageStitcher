## parallel image stitcher using ezSIFT

[![License][license-img]][license-url] [![Build Status](https://travis-ci.com/robertwgh/ezSIFT.svg?branch=master)](https://travis-ci.com/robertwgh/ezSIFT)

### How to build
#### Mac OS
Follow the following instructions:
```Bash
mkdir build
cd build
cmake ..
make
```
Then you can find the built binary under `build/bin` directory. Run the image_stitching code like this:

```bash
./image_stitching_seq img1.pgm img2.pgm 
```

### License

    Copyright 2013 Guohui Wang

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.


[license-url]: https://github.com/robertwgh/ezSIFT/blob/master/LICENSE
[license-img]: https://img.shields.io/badge/License-Apache%202.0-blue.svg
