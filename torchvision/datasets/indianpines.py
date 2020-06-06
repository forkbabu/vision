#content here.
import .utils import loadmat,select_small_cubic,sampling
from .vision import VisionDataset

class IndianPines(VisionDataset):
    """`Indian Pines  <http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Indian_Pines>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
              `IP` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        split (float,optional): Mentions the Validation Split. If split = 0.95 ,Training is 5% of the Total.
        
        PATCH_LENGTH (int,optional): Mentions the Window patch given by 2*PATCH_LENGTH + 1
    """
    base_folder = 'IP'
    
    ipath = "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"
    ifile = "Indian_pines_corrected.mat"
    imd5 = '66dbc9f4a9b7c9b1445f60a87b505101'
    
    lpath = "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_gt.mat"
    lfile = "Indian_pines_gt.mat"
    lmd5 = '9414943dac1d80faaa9165c8b460510c'
    
    def __init__(self, root, train=True,
                 download=False,split=0.95,PATCH_LENGTH=5):

        super(IndianPines, self).__init__(root)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        TOTAL_SIZE = 10249
        data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE,VALIDATION_SPLIT = loadmat(self.ipath,self.lpath,self.imd5,self.lmd5,TOTAL_SIZE,split)
	      data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))
	      gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]),)self.self.
	    self.  CLAself.SSES_NUM = max(gt)
	      img_channels = data_hsi.shape[2]
	      INPUT_DIMENSION = data_hsi.shape[2]
	      ALL_SIZE = data_hsi.shape[0] * data_hsi.shape[1]
	      VAL_SIZE = int(TRAIN_SIZE)
	      TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
	      data = preprocessing.scale(data)
	      data_ = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])
	      whole_data = data_
	      padded_data = np.lib.pad(whole_data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0,0)),'constant', constant_values=0)

        if self.train:
          train_indices, _ = sampling(VALIDATION_SPLIT, gt)
          _, total_indices = sampling(1, gt)
          TRAIN_SIZE = len(train_indices)
          
          #do stuff only for train
        else:
          _, test_indices = sampling(VALIDATION_SPLIT, gt)
          #do stuff only for test
        
        #common code
        
    def __getitem__(self, index):
      
      
    
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
            
        """
      y_train = gt[train_indices] - 1
      train_data = select_small_cubic(TRAIN_SIZE, train_indices, whole_data,PATCH_LENGTH,padded_data,INPUT_DIMENSION)
      x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION)
      x_train_tensor = np.squeeze(x_train,axis=1)
      y_train_tensor = np.squeeze(y_train,axis=1)
  
      return x_train_tensor[index,...],y_train_tensor[index,...]

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        ifilename, imd5 = self.ifile, self.imd5
        ifpath = os.path.join(root, self.base_folder, ifilename)  
        lfilename, lmd5 = self.lfile, self.lmd5
        lfpath = os.path.join(root, self.base_folder, lfilename)
        if check_integrity(lfpath, lmd5) and check_integrity(ifpath, imd5):
          return True
        return False

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_url(self.ipath, self.root, filename=self.ifile, md5=self.imd5)
        download_url(self.lpath, self.root, filename=self.lfile, md5=self.lmd5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
