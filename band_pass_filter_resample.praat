### Band-pass filter sound files ###

form Equalize SPL (sound pressure level) in dB
	real low_cuttoff(Hz) 300
	real high_cuttoff(Hz) 4000
	real new_sampling_freq(Hz) 8000
comment Give an optional suffix for all filenames (.wav will be added regardless):
	sentence Suffix
endform

#input folder selector
stimuli_folder$ = chooseDirectory$ ("Choose the folder containing sound files and textgrids")

#output folder selector
output_folder$ = chooseDirectory$ ("Select folder to write wave files to")

# This will need to be changed to "\" below for windowZ users
stimuli_folder$ = "'stimuli_folder$'" + "/" 
output_folder$ = "'output_folder$'" + "/" 
file_type$ = ".wav"

# List of all the sound files in the specified stimuli_folder:
Create Strings as file list... list 'stimuli_folder$'*'file_type$'
nstrings = Get number of strings

for i from 1 to nstrings

      select Strings list

      file$ = Get string... 'i'
	file$ = left$(file$,length(file$)-4)
	
      Read from file... 'stimuli_folder$''file$''file_type$'

	select Sound 'file$'

	Filter (pass Hann band)... low_cuttoff high_cuttoff 100

	Resample... new_sampling_freq 50

	Save as WAV file... 'output_folder$''file$''suffix$''file_type$'

endfor

select all
Remove
