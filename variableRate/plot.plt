set term postscript eps font "Times-Roman,24"  size 10in,5in
filename = "3"
set output "variableRate.eps"

set multiplot layout 2, 1

set title "Raw RSS"
unset key
set border 11 lw 2
set key right bottom
set ylabel "RSS (dBm)"

start = 1418842594.75337+43
#start = 0

set xr [0:60]
set yr [-90:-45]

plot "test".filename.".txt" using ($9-start):($2 != 127 ? $2 : 1/0) title '1st Channel' with lines ls 1 lc 1 lw 1 \
   , "test".filename.".txt" using ($9-start):($4 != 127 ? $4 : 1/0) title '2nd Channel' with lines ls 1 lc 2 lw 1 \
   , "test".filename.".txt" using ($9-start):($6 != 127 ? $6 : 1/0) title '3rd Channel' with lines ls 1 lc 3 lw 1 \
   , "test".filename.".txt" using ($9-start):($8 != 127 ? $8 : 1/0) title '4th Channel' with lines ls 1 lc 4 lw 1


set title "Breathing Estimation"
unset key
set border 11 lw 2
set key right bottom
set xlabel "Time (s)"
set ylabel "BPM"

set yr [10:25]

plot "test".filename."metronome.txt" using ($14-start):($2*0.5) title 'True BPM' with steps ls 1 lc 1 lw 1 \
   , "output".filename.".txt" using ($1-start):2 title 'Estimated BPM' with lines ls 1 lc 2 lw 1