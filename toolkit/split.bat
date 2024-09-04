@ECHO OFF
FOR /F %%i IN ('tshark -r Ursnif_TLS.pcap -T fields -e tcp.stream ^| sort /unique') DO (
::ECHO %%i 
	tshark -r Ursnif_TLS.pcap -Y "tcp.stream eq %%i" -w "stream-%%i.pcap" 
)