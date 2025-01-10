form Extraction dossier audio
    text dossier_audio ../data/audio_original/
endform

generatedFolder$ = "../data/audio_decoupe/"
createDirectory: generatedFolder$

regex_chemin_vers_wav$ = dossier_audio$ + "*.wav"
filesList = Create Strings as file list: "fileslist", regex_chemin_vers_wav$
nfiles = Get number of strings

for ifile to nfiles
    selectObject:filesList
	nomFichierWav$ = Get string: ifile
	nomFichierBase$ = nomFichierWav$ - ".wav"
    generatedFolderLocuteur$ = generatedFolder$ + nomFichierBase$
    createDirectory: generatedFolderLocuteur$

	chemin_absolu_vers_son$ = dossier_audio$ + nomFichierWav$
	son = Read from file: chemin_absolu_vers_son$

	#---------------------------------------------------------------
	selectObject: son
    tDeb = Get start time
    tEnd = Get end time
    tAdd = 0.01
    tFin = 0.05
    num = 0
    
    while tFin < tEnd
        duree = tFin - tDeb
        num += 1
        selectObject: son
        sonExtrait = Extract part: tDeb, tFin, "rectangular", 1, "no"
        folderSounds$ = generatedFolderLocuteur$ + "/" + nomFichierBase$ +"-" + string$(num) + "_" + string$(tDeb) + "_" + string$(tFin) + "_" + string$(duree) + ".wav"
        #pause 'folderSounds$'
        Save as WAV file: folderSounds$
        select sonExtrait
        Remove

        tDeb = tDeb + tAdd
        tFin = tFin + tAdd
    endwhile
    if tFin > tEnd
        duree = tFin - tDeb
        tFin = tEnd
        num += 1

        selectObject: son
        sonExtrait = Extract part: tDeb, tFin, "rectangular", 1, "no"
        folderSounds$ = generatedFolderLocuteur$ + "/" + nomFichierBase$ +"-" + string$(num) + "_" + string$(tDeb) + "_" + string$(tFin) + "_" + string$(duree) + ".wav"
        Save as WAV file: folderSounds$
        select sonExtrait
        Remove
    endif

	select all
	minus 'filesList'
	Remove
endfor



