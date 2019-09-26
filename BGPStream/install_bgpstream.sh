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


### wandio
cd ~/src
curl -O https://research.wand.net.nz/software/wandio/wandio-4.0.0.tar.gz
tar zxf wandio-4.0.0.tar.gz
cd wandio-4.0.0/
sudo ./configure
sudo make
sudo make install


### bgpstream-core
cd ~/src
#curl -O http://bgpstream.caida.org/bundles/caidabgpstreamwebhomepage/dists/bgpstream-1.2.3.tar.gz
wget http://bgpstream.caida.org/bundles/caidabgpstreamwebhomepage/dists/bgpstream-1.2.3.tar.gz -O bgpstream-1.2.3.tar.gz
tar zxf bgpstream-1.2.3.tar.gz
cd bgpstream-1.2.3
sudo ./configure
sudo make
sudo make install


### pybgpstream
sudo -H pip3.7 install pybgpstream
