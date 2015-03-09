set term postscript eps font "Times-Roman,24"  size 10in,5in
filename = "breathing_rf_face_up_omni"
set output "Omni.eps"

set multiplot layout 2, 1

set title "Raw RSS"
unset key
set border 11 lw 2
set key left bottom
set xlabel "Time (s)"
set ylabel "RSS (dBm)"

start = 1417739310.08558
#start = 0

#set xr [0:10]

plot filename.".txt" using ($9-start):($2 != 127 ? $2 : 1/0) title '1st Channel' with lines ls 1 lc 1 lw 1 \
   , filename.".txt" using ($9-start):($4 != 127 ? $4 : 1/0) title '2nd Channel' with lines ls 1 lc 2 lw 1 \
   , filename.".txt" using ($9-start):($6 != 127 ? $6 : 1/0) title '3rd Channel' with lines ls 1 lc 3 lw 1 \
   , filename.".txt" using ($9-start):($8 != 127 ? $8 : 1/0) title '4th Channel' with lines ls 1 lc 4 lw 1


set title "Breathing Estimation"
unset key
set ylabel "BPM"
plot "output_face_up_omni.txt" using ($1-start):2 title 'BPM' with lines ls 1 lc 5 lw 1 
