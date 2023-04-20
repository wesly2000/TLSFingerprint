:: If the extension of a file is .pcap, we first filter the TLS packet
:: by "ssl.record.version >= 0x0301"
:: Then we convert the file to .json
@echo off
SETLOCAL EnableDelayedExpansion
set _extend=json

for /f "tokens=4" %%i in ('dir -s "*.pcap"') do (
    set _file=%%i
    if !_file:~-4!==pcap (
        tshark -T json -Y "ssl.record.version>=0x0301" -r "!_file!">"!_file:~0,-4!%_extend%"
    )
)
