#!/bin/bash

date

### compulsory ###
ncore=1
tprfile=input.sA.tpr
gmx=`which gmx_mpi`
script=/home/npedrani@iit.local/Desktop/Phd_main_Projects/Hyper-TICA/Alanine/script/bck.meup.sh

### optional ###
nsteps=$[500*1000*5] #last is ns
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
#before run this execute cp ../alanine* .
#mpi_cmd="$gmx mdrun -s $tprfile -deffnm $filename $plumedfile $ntomp $nsteps -cpi $filename"
mpi_cmd="$gmx mdrun -s $tprfile -deffnm $filename $plumedfile $ntomp $nsteps"
submit="time mpirun -np $ncore ${mpi_cmd} -pin on -pinoffset 0 -pinstride 1"
#submit="time mpirun -np $ncore ${mpi_cmd} -pin off" #change this when submitting to a cluster

### execute ###
bash ${script} -i $outfile
bash ${script} -i ${filename}* > $outfile
echo -e "\n$submit &>> $outfile"
eval "$submit &>> $outfile"
[ -z "$extra_cmd" ] || eval $extra_cmd

date
