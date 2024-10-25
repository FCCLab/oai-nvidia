# User guide

### Update SDR MAC address (front of the sdrv4) in file `cardmap`

<img src="imgs\SDR_MAC_ADDR.jpg" alt="screenshot of user guide" width="50%">

```
SDR_BASE 0 7 80:09:02:1A:0D:BF 0 0 5
```


### Copy configuration tar.gz file to 
```
/lsu/cfg/archive.www
```

### UDG

CoreSIM (10.5.6.1) <--> (10.5.6.14) UDG on eLSU

* Update `csv` files
    * ipv4_server.csv
        ```
        Value,1,0,192.168.60.14,255.255.255.0,50302,21
        Value,2,1,10.5.6.14,255.255.255.0,50101,9
        ```
    * VPN
        * ipv4_vpn_address.csv
            ```
            Value,1,1,10.5.6.14,255.255.255.0
            ```
        * ipv4_vpn_routing.csv
            ```
            Value,1,1,10.5.6.1
            ```
    * OOB
        * Oob-UdgSrv_over_IPv4.csv
            ```
            Value,1,0,192.168.60.14,255.255.255.0,50302,21
            ```

* Card map
    ```
    PPU_CAPABILITY  STK_U-Plane0.1 FUNC:UDG_OOB 1 6 10.5.6.0 24 192.168.60.0
    IP_ALIAS     STK_U-Plane0.2 7 2 1 4 10.5.6.14 24
    ```

* Update `set_gi` with `10.5.6.1` where `10.5.6.1` is the IP address of N6 interface of CoreSIM
    ```
    route add default 10.5.6.1
    ```

