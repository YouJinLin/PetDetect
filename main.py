import cv2
import os
from random import randint
import torch
from DataDetect import Detect

def main():
    file_dir = r'D:\python\Petsource\test\test'
    filelist = os.listdir(file_dir)
    total_num = len(filelist)
    idx = randint(0, total_num)
    imgPath = filelist[idx]
    path = os.path.join(file_dir, imgPath)
    
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #print(img)
    img_cv_Tensor = torch.tensor(img)
    img_cv_Tensor = torch.unsqueeze(img_cv_Tensor, dim=0)
    img_cv_Tensor = img_cv_Tensor.to(float)
    
    result, _ = Detect(img_cv_Tensor)
    print(result)

    cv2.imshow('img', img)
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()