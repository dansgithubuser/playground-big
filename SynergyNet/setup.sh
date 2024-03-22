cd SynergyNet  # https://github.com/choyingw/SynergyNet/commit/5fe2e72c441fcd00f5720776b6f1e3d956296159
unzip aflw2000_data.zip
unzip 3dmm_data.zip
cd Sim3DR
./build_sim3dr.sh
cd ../FaceBoxes
./build_cpu_nms.sh
