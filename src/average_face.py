from pathlib import Path
import json
import cv2
import numpy as np

results_path = Path("..", "Results", "UNIFESP")

results = {
    'com dor': {
        'img': None,
        'count': 0
    },
    'sem dor': {
        'img': None,
        'count': 0
    },
    'sem filtro': {
        'img': None,
        'count': 0
    }
}

for directory in results_path.glob('*'):
    with open(directory / 'face_data.json', 'r') as f:
        label = json.load(f)['label']
    img = cv2.imread(str(directory / 'bbox_crop-affine.png'), 0)

    # try: results[label]['img'] = (results[label]['img'].astype(np.uint64) + img.astype(np.uint64))
    # except: results[label]['img'] = img
    
    # try: results['sem filtro']['img'] = (results['sem filtro']['img'].astype(np.uint64) + img.astype(np.uint64))
    # except: results['sem filtro']['img'] = img
    
    try: results['sem filtro']['img'] = np.dstack( (results['sem filtro']['img'], img ) )
    except: results['sem filtro']['img'] = img

    # results[label]['count'] += 1
    # results['sem filtro']['count'] += 1

print(results['sem filtro']['img'].shape)
results['sem filtro']['img'] = np.sort(results['sem filtro']['img'], axis=2)
results['sem filtro']['img'] = np.median(results['sem filtro']['img'], axis=2)
cv2.imwrite(str(results_path.parent / ('face_media-sem_filtro.jpg')), results['sem filtro']['img'].astype(np.uint8))

# for key, value in results.items():
#     cv2.imwrite(str(results_path.parent / ('face_media-'+key.replace(' ','_')+'.jpg')), (value['img'] / value['count']).astype(np.uint8))
