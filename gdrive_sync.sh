# to run from local machine while inside text/bias/bbq folder

rclone sync gdrive:/bbq/final_templates/final_templates_without_WORDS ./templates_es --progress
rclone sync gdrive:/bbq/final_templates/vocabulary_es.xlsx ./templates_es --progress
rclone sync gdrive:/bbq/final_templates/vocabulary_proper_names_es.xlsx ./templates_es --progress
