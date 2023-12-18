cd ~/synthesis

sbatch run_qaoa_nono.sh
sbatch run_synth_nono.sh
sbatch run_inst_nono.sh

sbatch run_qaoa.sh
sbatch run_synth.sh
sbatch run_inst.sh

sbatch run_qaoa_hard.sh
sbatch run_synth_hard.sh
sbatch run_inst_hard.sh

