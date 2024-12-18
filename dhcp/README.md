# DHCP

```
sudo ifconfig ens12f0 192.168.1.1/24
docker run -it --rm --init --net host -v "$(pwd)/data":/data networkboot/dhcpd ens12f0
```
