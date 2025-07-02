# Liteon 02

```
➜  ~ ssh user@10.1.7.194         
The authenticity of host '10.1.7.194 (10.1.7.194)' can't be established.
ED25519 key fingerprint is SHA256:bSUwQIkP7H/elP09jEnh0rZ8E7K5XoSl8k/h31W4VGI.
This key is not known by any other names
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
Warning: Permanently added '10.1.7.194' (ED25519) to the list of known hosts.
user@10.1.7.194's password: 
Welcome to Liteon ORU Wed Jun 25 10:15:31 UTC 2025
Please enter help
> 
> 
> enable
Enter Password: 
Auto exit privileged commands in 300 Seconds
# 
# 
# 
# show running-config 
Band Width = 100000000
Center Frequency = 3425010000
Compression Bit = 9
Control and User Plane vlan = 3
M Plane vlan = 0
default gateway = 10.1.7.1
dpd mode : Enable
DU MAC Address = 9c63c0a70832
phase compensation mode : Enable
RX attenuation = 14
TX attenuation = 0
subcarrier spacing = 1
rj45_vlan_ip = 10.102.131.61
SFP_vlan_ip = 10.102.131.62
SFP_non_vlan_static_ip = 10.101.131.194
prach eAxC-id port 0, 1, 2, 3 = 0x0000, 0x0001, 0x0002, 0x0003
slotid = 0x00000001
jumboframe = 0x00000001
sync source : PTP
# 

# show eth-info 
eth0      Link encap:Ethernet  HWaddr e8:c7:4f:25:7f:7c  
          inet addr:10.1.7.194  Bcast:10.1.7.255  Mask:255.255.255.0
          inet6 addr: fe80::eac7:4fff:fe25:7f7c/64 Scope:Link
          UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
          RX packets:663 errors:0 dropped:0 overruns:0 frame:0
          TX packets:262 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000 
          RX bytes:68980 (67.3 KiB)  TX bytes:37563 (36.6 KiB)
          Interrupt:29 

eth1      Link encap:Ethernet  HWaddr e8:c7:4f:25:89:40  
          inet addr:10.101.131.194  Bcast:10.101.131.255  Mask:255.255.255.0
          inet6 addr: fe80::eac7:4fff:fe25:8940/64 Scope:Link
          UP BROADCAST RUNNING  MTU:1500  Metric:1
          RX packets:2128780 errors:0 dropped:386595 overruns:0 frame:0
          TX packets:74001 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000 
          RX bytes:2501727660 (2.3 GiB)  TX bytes:4513722 (4.3 MiB)

lo        Link encap:Local Loopback  
          inet addr:127.0.0.1  Mask:255.0.0.0
          inet6 addr: ::1/128 Scope:Host
          UP LOOPBACK RUNNING  MTU:65536  Metric:1
          RX packets:258 errors:0 dropped:0 overruns:0 frame:0
          TX packets:258 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000 
          RX bytes:22704 (22.1 KiB)  TX bytes:22704 (22.1 KiB)

# 
```


```
ssh user@10.1.7.194
user/user
enable
liteon166
show sync-trace
```

```
# show sync-trace 
Command Processed successfully.
Sync status/state: NOT SYNCHRONIZED/SYNCHRONIZING
```