sudo ip addr add 192.168.1.1/24 dev aerial0.d9
docker run -it --rm --init --net host -v "$(pwd)/data_liteon":/data networkboot/dhcpd aerial0.d9
