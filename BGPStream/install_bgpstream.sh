cd ~
mkdir ~/src

### libbz2
sudo apt-get install libbz2-1.0 libbz2-dev


### zlib
cd ~/src
wget http://www.zlib.net/zlib-1.2.11.tar.gz -O zlib-1.2.11.tar.gz
tar -zxf zlib-1.2.11.tar.gz
cd zlib-1.2.11/
sudo ./configure
sudo make
sudo make install


### libcurl
cd ~/src
wget https://curl.haxx.se/download/curl-7.65.3.tar.gz -O curl-7.65.3.tar.gz
tar -zxf curl-7.65.3.tar.gz
cd curl-7.65.3/
sudo ./configure
sudo make
sudo make install
sudo ldconfig

### wandio
cd ~/src/
curl -LO https://research.wand.net.nz/software/wandio/wandio-4.2.0.tar.gz
tar zxf wandio-4.2.0.tar.gz
cd wandio-4.2.0/
./configure
make
make install
sudo ldconfig


### bgpstream-core
cd ~/src/
curl -LO https://github.com/CAIDA/libbgpstream/releases/download/v2.0-rc2/libbgpstream-2.0.0-rc2.tar.gz
tar zxf libbgpstream-2.0.0-rc2.tar.gz
cd libbgpstream-2.0.0
sudo ./configure --without-kafka
sudo make
sudo make install


### pybgpstream
sudo apt-get install python3-apt
sudo apt-get install libpython3.7-dev
sudo -H python3.7 -m pip install pybgpstream
