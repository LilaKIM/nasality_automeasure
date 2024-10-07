form Extraction dossier audio
    text dossier_audio /media/lila/T7 Shield1/ptsvox/test_praat/
    real nb_extrait 10
endform

generatedFolder$ = "../data/audio_4s/"
createDirectory: generatedFolder$

regex_chemin_vers_wav$ = dossier_audio$ + "*.wav"
filesList = Create Strings as file list: "fileslist", regex_chemin_vers_wav$
nfiles = Get number of strings

for ifile to nfiles
    selectObject:filesList
	nomFichierWav$ = Get string: ifile
	nomFichierBase$ = nomFichierWav$ - ".wav"

	chemin_absolu_vers_son$ = dossier_audio$ + nomFichierWav$
	son = Read from file: chemin_absolu_vers_son$

	#---------------------------------------------------------------
	selectObject: son
    tEnd = Get end time
    tEndInt = round (tEnd - 5)
    # pause 'tEnd', 'tEndInt'
    tAdd = 4
    ct = 0
    temp = 0
    repeat
        tDeb = randomInteger (1,tEndInt)
        if temp <> tDeb
            tFin = tDeb + tAdd
            # pause 'tDeb', 'tFin' 'tEndInt', 'tEnd'
            ct += 1
            temp = tDeb
            duree = tFin - tDeb
            selectObject: son
            sonExtrait = Extract part: tDeb, tFin, "rectangular", 1, "no"
            folderSounds$ = generatedFolder$ + "/" + nomFichierBase$ +"-" + string$(ct) + "_" + string$(tDeb) + "_" + string$(tFin)+ ".wav"
            Save as WAV file: folderSounds$
            select sonExtrait
            Remove
        endif
    until ct = nb_extrait

	select all
	minus 'filesList'
	Remove
endfor



