#!/bin/bash

date

### compulsory ###
ncore=1
tprfile=input.sB.tpr
gmx=`which gmx_mpi`

### optional ###
nsteps=$[500*1000*100] #last is ns
ntomp=2
#maxh=1:00 #h:min
filename=alanine
plumedfile=plumed.dat
extra_cmd=""

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
mpi_cmd="$gmx mdrun -s $tprfile -deffnm $filename $plumedfile $ntomp $nsteps"
submit="time mpirun -np $ncore ${mpi_cmd} -pin off" #change this when submitting to a cluster
#submit="time ${mpi_cmd} -pin off"
### execute ###
../script/./bck.meup.sh -i $outfile
../script/./bck.meup.sh -i ${filename}* > $outfile
echo -e "\n$submit &>> $outfile"
eval "$submit &>> $outfile"
[ -z "$extra_cmd" ] || eval $extra_cmd

date
