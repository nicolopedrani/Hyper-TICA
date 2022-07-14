#!/bin/bash

date

### compulsory ###
ncore=1
tprfile=input.sB.tpr
gmx=`which gmx_mpi`
script=/home/npedrani@iit.local/Desktop/Phd_main_Projects/Hyper-TICA/Alanine/script/bck.meup.sh
pin_offset=4
cpi_state=false

### optional ###
ns=4.5
nsteps=$(echo "500*1000*$ns" | bc | awk '{ printf("%.0f\n",$1) '})
ntomp=2
#maxh=1:00 #h:min
filename=alanine
restartfile=$filename*.cpt
checkpointfile=state.cpt
plumedfile=plumed.dat
extra_cmd=""
gpu_id=1

echo gromacs for $nsteps steps

### setup ###
[ -z "$filename" ]  && filename=simulation
outfile=${filename}.out
[ -z "$plumedfile" ] || plumedfile="-plumed $plumedfile"
[ -z "$ntomp" ] || ntomp="-ntomp $ntomp"
[ -z "$nsteps" ] || nsteps="-nsteps $nsteps"
if [ ! -z "$maxh" ]
then
  maxh=`python <<< "print('%g'%(${maxh%:*}+${maxh#*:}/60))"`
  maxh="-maxh $maxh"
fi

### commands ###
#mpi_cmd="$gmx mdrun -s $tprfile -deffnm $filename $plumedfile $ntomp $nsteps $maxh"

if ${cpi_state}
then
  mpi_cmd="$gmx mdrun -s $tprfile -cpo $checkpointfile -cpi state -noappend -deffnm $filename $plumedfile $ntomp $nsteps"
else
  mpi_cmd="$gmx mdrun -s $tprfile -cpo $checkpointfile -deffnm $filename $plumedfile $ntomp $nsteps"
fi

submit="time mpirun -np $ncore ${mpi_cmd} -pin on -pinoffset $pin_offset -pinstride 1 -gpu_id $gpu_id -reseed 1"

### execute ###
# vorrei eseguirlo ma mi da problemi con il restart.. 
#bash ${script} -i $outfile
#bash ${script} -i ${filename}* > $outfile
echo -e "\n$submit &>> $outfile"
eval "$submit &>> $outfile"
[ -z "$extra_cmd" ] || eval $extra_cmd

date
