set term postscript eps font "Times-Roman,22" size 4.5in,2.7in
filename = "breathing_rf_face_up_omni"
set output "orientationComparison.eps"

set border 11 lw 2

set xr [0:60]
set yr [14:16]
set ytics 0.2

set ylabel "BPM"
set xlabel "Time (s)"

start1 = 1417733038.54534+120
start2 = 1417735254.32042+120
start3 = 1417734139.83089+120

f(x) = 15
plot f(x) title "True BPM" with lines lc 4 lw 5 lt 1 \
   , "output_chest_up.txt" using ($1-start1):2 title 'Face up' with lines ls 2 lc 1 lw 5 \
   , "output_face_down.txt" using ($1-start2):2 title 'Face down' with lines ls 3 lc 2 lw 5 \
   , "output_side_up.txt" using ($1-start3):2 title 'Side' with lines ls 4 lc 3 lw 5 
