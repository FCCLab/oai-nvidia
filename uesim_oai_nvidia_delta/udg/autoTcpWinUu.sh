SOCK=/sock2 on -f ppu0-9 sysctl -w net.inet.tcp.recvspace=8000000
SOCK=/sock2 on -f ppu0-19 sysctl -w net.inet.tcp.recvspace=8000000
SOCK=/sock2 on -f ppu0-9 sysctl -w net.inet.tcp.sendspace=8000000
SOCK=/sock2 on -f ppu0-19 sysctl -w net.inet.tcp.sendspace=8000000
SOCK=/sock2 on -f ppu0-9 sysctl -w net.inet.tcp.recvbuf_auto=0
SOCK=/sock2 on -f ppu0-9 sysctl -w net.inet.tcp.recvbuf_inc=150000
SOCK=/sock2 on -f ppu0-9 sysctl -w net.inet.tcp.recvbuf_max=10000000
SOCK=/sock2 on -f ppu0-9 sysctl -w net.inet.tcp.sendbuf_auto=0
SOCK=/sock2 on -f ppu0-9 sysctl -w net.inet.tcp.sendbuf_inc=150000
SOCK=/sock2 on -f ppu0-9 sysctl -w net.inet.tcp.sendbuf_max=10000000
SOCK=/sock2 on -f ppu0-9 sysctl -w kern.sbmax=22000000
SOCK=/sock2 on -f ppu0-19 sysctl -w kern.sbmax=22000000
SOCK=/sock2 on -f ppu0-9 sysctl -w kern.mbuf.nmbclusters=500000
SOCK=/sock2 on -f ppu0-19 sysctl -w kern.mbuf.nmbclusters=500000
SOCK=/sock2 on -f ppu0-9 sysctl -w net.inet.ip.ifq.maxlen=2048
SOCK=/sock2 on -f ppu0-9 ifconfig en1 mtu 1350
SOCK=/sock2 on -f ppu0-19 ifconfig en1 mtu 1350
