set term postscript eps font "Times-Roman,24"  size 10in,5in
filename = "breathing_2"
set output filename.".eps"

set border 11 lw 2
set key right top
set xlabel "Time (s)"
set ylabel "RSS (dBm)"

start = 1409956513.96017
#start = 0

set xr [0:60]
set yr [-70:-50]

plot filename.".txt" using ($9-start):($2 != 127 ? $2 : 1/0) title '1st Channel' with lines ls 1 lc 1 lw 1 \
   , filename.".txt" using ($9-start):($4 != 127 ? $4 : 1/0) title '2nd Channel' with lines ls 1 lc 2 lw 1 \
   , filename.".txt" using ($9-start):($6 != 127 ? $6 : 1/0) title '3rd Channel' with lines ls 1 lc 3 lw 1 \
   , filename.".txt" using ($9-start):($8 != 127 ? $8 : 1/0) title '4th Channel' with lines ls 1 lc 4 lw 1
