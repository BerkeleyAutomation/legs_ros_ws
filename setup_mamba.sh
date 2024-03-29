#!/usr/bin/expect -f

# Set timeout to 1 second between each newline
set timeout 1

# Spawn the shell script
spawn bash /root/Mambaforge-Linux-x86_64.sh

# Loop to send newline character 20 times
for {set i 0} {$i < 170} {incr i} {
    send "\n"
    #expect "continue"  # Wait for the next prompt before sending the next newline
}
send "yes\n\n"
# Wait for the script to finish
expect eof
