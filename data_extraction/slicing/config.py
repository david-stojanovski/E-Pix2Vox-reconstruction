from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.DATA_IN = edict()
__C.DATA_IN.DATA_FOLDER = r'/folder/containing/binary/meshes/'
__C.DATA_IN.SAVE_FOLDER = r'/folder/to/save/slices/to'

#
# Dataset Config
#
__C.LABELS = edict()
__C.LABELS.LABEL_NAME = 'elemTag'
__C.LABELS.LV = 1
__C.LABELS.RV = 2
__C.LABELS.LA = 3
__C.LABELS.RA = 4
__C.LABELS.AORTA = 5
__C.LABELS.MITRAL_VALVE = 7
__C.LABELS.AORTIC_VALVE = 9
__C.LABELS.PULMONARY_VALVE = 10

__C.DATA_OUT = edict()
__C.DATA_OUT.SELECTED_VIEWS = ['rv_inflow', 'lv_plax', 'psax_aortic', 'psax_mv',
                               'psax_pm', 'psax_lower', 'a4c', 'a5c', 'a2c']
__C.DATA_OUT.SAVE_IMG_RESOLUTION = (224, 224)  # Hasn't been verified for aspect ratios other than 1:1
__C.DATA_OUT.SAVE_BCKGD_CLR = 'black'
__C.DATA_OUT.FANCY_PLOT = False

__C.PARAMETERS = edict()
__C.PARAMETERS.SUBSAMPLE_FACTOR = 20  # How much to subsample mesh for the low resolution, fast calculations
__C.PARAMETERS.THRESHOLD_PERCENTAGE = 0.7  # Percentage of mesh to remove in order to speed up finding LV apex
__C.PARAMETERS.NUM_WORKERS = 8  # Number of CPU cores to use for multiprocessing

# The following are the labels used in the generation of the meshes from https://zenodo.org/record/4506930#.Yt_sZtLMJ1M
# 01. LV myocardium (endo + epi)
# 02. RV myocardium (endo + epi)
# 03. LA myocardium (endo + epi)
# 04. RA myocardium (endo + epi)
# 05. Aorta
# 06. Pulmonary artery
# 07. Mitral valve
# 08. Triscupid valve
# 09. Aortic valve
# 10. Pulmonary valve
# 11. Appendage
# 12. Left superior pulmonary vein
# 13. Left inferior pulmonary vein
# 14. Right inferior pulmonary vein
# 15. Right superior pulmonary vein
# 16. Superior vena cava
# 17. Inferior vena cava
# 18. Appendage border
# 19. Right inferior pulmonary vein border
# 20. Left inferior pulmonary vein border
# 21. Left superior pulmonary vein border
# 22. Right superior pulmonary vein border
# 23. Superior vena cava border
# 24. Inferior vena cava border
