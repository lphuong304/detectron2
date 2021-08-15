# Copyright (c) Facebook, Inc. and its affiliates.
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch
import numpy as np
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        MetadataCatalog.get("dataset_test12").thing_classes = ['product', 'AFC bo bit tet 200g', 'Aquafina 500ml', 'Banh Choco-pie hop 198g -6cai-', 'Banh Choco-pie hop 66g -2cai-', 'Banh Cookie Andersen 100g', 'Banh Cream Kem dau 85g', 'Banh Cream socola 85g', 'Banh Cream socola kem socola 85g', 'Banh Hello Panda Matcha 50g', 'Banh Kitkat socola 35g', 'Banh Nabati socola 52g', 'Banh Omeli Chocolate Pie hop 300g', 'Banh Oreo Kem Lanh Viet Quat 133g', 'Banh Pho Mai Cal Cheese 53g', 'Banh Pillows Nhan Socola 100g', 'Banh Pocky Chocolate 40g', 'Banh Pocky Cookies Cream Taste 40g', 'Banh Pocky Huong dau 40g', 'Banh Pocky Matcha 40g', 'Banh StrawBerry bauli 45g', 'Banh cracker khoai tay Omeli hop 128g', 'Banh cracker man vung dua Omeli hop 128g', 'Banh gao Shouyu mat ong Ichi 100g', 'Banh gao pho mai bap One One 118g', 'Banh gau nhan kem sua Meiji Hello Panda hop 50g', 'Banh mi tuoi 6 mui cha bong Kinh Do o 80g', 'Banh phu socola vi bo sua Solite hop 280g', 'Banh que Mix huong vi ga cay goi 60g', 'Banh que Mix vi ot cay goi 60g', 'Banh que vi kem cam Cosy goi 132g', 'Banh que vi kem la dua Cosy 132g', 'Banh que vi kem so co la Cosy goi 132g', 'Banh quy hat socola yen mach Cosy Original goi 80g', 'Banh quy sua cosy marie 432g', 'Banh xop pho mai Richeese Ahh Triple hop 160g', 'Bia Sai Gon Lager 330ml', 'Bich Keo MaLai Vi Khoai Mon Taro 67g', 'Binh cach nhiet co khoa T', 'Binh giu nhiet lovely 500ml', 'Bo 2 cai luoi dao cao rau 2 luoi Gillette Vector', 'Bo 4 cai luoi dao cao rau 2 luoi Gillette Vector', 'Bo thuc vat Meizan hu 200g', 'Bot banh ran Ajimoto vi truyen thong 200g', 'Bot bap Tai Ky goi 150g', 'Bot cao rau Gillette huong chanh 175g', 'Bot chien gion Aji-Quick goi 150g', 'Bot chien gion Meizan 150g do', 'Bot chien gion Meizan 150g xanh', 'Bot chien tam uop cay Tai Ky goi 60g', 'Bot mi da dung Meizan cao cap goi 500g', 'Bot mi da dung Tai Ky goi 500g', 'Bot nang Tai Ky goi 1kg', 'Bot nang da dung Meizan goi 400g', 'Bot ngot Ajinomoto goi 400g', 'Bot thach khuc bach huong socola Dragon hop 106g', 'C2 Tao tra xanh 230ml', 'C2 huong chanh chai 455ml', 'C2 huong tra dao chai 455ml', 'Ca nuc xot ca 3 Co Gai hop 190g', 'Ca phe sua G7 3 in 1 288g -18 goi x 16g-', 'Ca phe sua G7 gu manh X2 300g -12 goi x 25g-', 'Ca sot ca chua HiChef 155g', 'Chao suon Vifon 70g', 'Chao yen Vifon 50g', 'Dao cao rau 2 luoi Gillette Vector', 'Dao cao rau luoi don Gillette Super Thin', 'Dao roc giay lon 18mm TTH', 'Dau goi Dove ngan gay rung 621ml', 'Dau goi Dove phuc hoi hu ton 874ml', 'Dau goi Lifebuoy toc mem muot 621ml', 'Dau goi TRESemme Salon Rebond giam rung toc 825ml', 'Dau goi huong nuoc hoa Romano Classic toc chac khoe 380g', 'Dau goi sach gau Clear Men Cool Sport bac ha 874ml', 'Dau goi sach gau Clear mat lanh bac ha 631ml', 'Dau xa Sunsilk mem muot dieu ki 327ml', 'Fanta Soda Kem lon 330ml', 'Gel rua tay kho Lifebuoy bao ve vuot troi chai 100ml', 'Hat dieu TuanDat 200g', 'Hat nem 25 duong chat Chinsu 400g', 'Hat nem thit than Knorr goi 400g', 'Hop Sua Yoho SoyMilk 190ml', 'Hop banh Hura cuon kem bo sua 360g', 'Hop banh Nabati cheese -20goix17g-', 'Kem dac co duong Ngoi sao Phuong Nam xanh la lon 380g', 'Kem danh rang Closeup Lua - Bang 180g', 'Kem danh rang Closeup tinh the bang tuyet 180g', 'Kem danh rang Closeup trang rang dua than hoat tinh 180g', 'Kem danh rang Colate MaxFresh tinh the 180g', 'Kem danh rang Colate Total than hoat tinh 190g', 'Kem danh rang Colgate MaxFresh than tre 200g', 'Kem danh rang Colgate MaxFresh tra xanh 200g', 'Kem danh rang Colgate Naturals chanh va lo hoi', 'Kem danh rang Colgate Optic White Volcanic Mineral 100g', 'Kem danh rang PS than hoat tinh 180g', 'Kem danh rang PS tinh hoa thien nhien 180g', 'Kem danh rang Sensodyne Coll Gel 100g', 'Kem danh rang Sensodyne Fresh Mint giam e buot 247 100g', 'Kem danh rang Sensodyne trang rang tu nhien 100g', 'Kem danh rnag PS Tra Xanh 100g', 'Kem tay da nang Cif huong chanh chai 690g', 'Keo dua Yen Huong 400g', 'Keo gung Migita goi 70g', 'Keo lon propaper', 'Keo me Tamarin 35g', 'Khan Uot Softify 20 to cao cap', 'Khan giay an PREMIER VinaTissue 1 lop goi 100 to', 'Khan giay an Watersilk 1 lop goi 100 to', 'Khan giay uot BabyCare 100 to', 'Khan giay uot Bobby 100 to', 'Kim go mesa', 'Lo 2 bang keo 18mmx18mm', 'Loc 4 hop sua bap non LiF 180ml', 'Loc 4 hop sua chua uong vi luu YoMost 170ml', 'Loc 4 hop sua dau nanh oc cho Vinamilk 180ml', 'Loc 4 hop sua dutch lady co duong', 'Loc 4 hop sua yomost dau', 'Loc 4 hop thuc uong lua mach Milo Active Go 180ml', 'Loc 4 hop yomost huong cam', 'Loc 4 hop yomost huong viet quat', 'Lon Cafe Sua HighLand 185ml', 'Lon Fanta Viet Quat lon 333ml', 'Lon Lavie Chanh Bac Ha 330ml', 'Lon NesCafe Latte 180ml', 'Mi A-One vi bo xao 85g', 'Mi De Nhat huong vi tom chua cay goi 81g', 'Mi Hao Hao Sa te hanh 75g', 'Mi Hao Hao suon heo toi phi goi 73g', 'Mi Hao Hao tom chua cay goi 75g', 'Mi Lau Thai tom goi 80g', 'Mi ly omachi suon ham 113g', 'Mi ly omachi tom chua cay 113g', 'Mi tron Omachi xot Spaghetti goi 91g', 'Mi xao Hao Hao tom xao chua ngot goi 75g', 'Mien Phu Huong Ga 53g', 'Mien Phu Huong Huong Vi Lau Thai Tom 60g', 'Mirinda Da me Lon 330ml', 'Mirinda soda kem 33ml', 'Mirinda vi cam 333ml', 'Nuoc cot dua Vietcoco hop 400ml', 'Nuoc giat xa Arota khang khuan khu mui cho tre chai 3kg', 'Nuoc lau kinh CIF sieu nhanh chai 520ml', 'Nuoc lau san SUnlight huong hoa dien vy chai 1kg', 'Nuoc mam nam ngu 500g', 'Nuoc ngot COca Cola 320ml', 'Nuoc ngot Pepsi khong calo vi chanh 330ml', 'Nuoc rua chen Sunlight Chanh 100 chiet xuat chanh tuoi chai 1450ml', 'Nuoc rua chen Sunlight Chanh 100 chiet xuat chanh tuoi chai 386ml', 'Nuoc rua chen Sunlight Extra chanh la bac ha chai 386ml', 'Nuoc rua chen Sunlight Extra tra xanh matcha Nhat Ban chai 386ml', 'Nuoc suc mieng Colgate Plax Peppermint Fresh 750ml', 'Nuoc tay bon cau nha tam VIM diet khuan 500ml', 'Nuoc tay da nang CiF huong chanh chai 520ml', 'Nuoc tay quan ao mau AXO huong hoa dao 400ml', 'Nuoc tuong dau nanh dam dac Maggi chai 300ml', 'Nuoc tuong tam thai tu 500g', 'Nuoc tuong toi ot Chinsu chai 250ml', 'Nuoc xit tay da nang Sowa chai 475ml', 'Nuoc xot me rang Kewpie chai 210ml', 'Pate gan dong hop Ha Long hop 150g', 'Peke Potato Vi Vit Quay chips 80g', 'Pepsi lon xanh 330ml', 'Pin energizer aa2', 'Pin energizer aaa4', 'Sap thom Glade huong hoa oai huong 180g', 'Snack Indochip 40g', 'Snack bi do vi bo nuong oishi 40g', 'Snack hanh Oishi xanh 40g', 'Snack khoai tay Slide vi nguyen ban lon 160g', 'Snack khoai tay vi thom cay Slide lon 75g', 'Snack pho mat mieng Oishi goi 40g', 'Snack phong muc Oishi Indo Ships goi 40g', 'Sot lau Thai Cholimex chai 280g', 'Sot mayonnaise Aji-mayo Ajinomoto chai 260g', 'Sua Bich Fami Vi Dau Do Nep Cam 200ml', 'Sua bich dutch lady huong dau 220ml', 'Sua bich dutch lady huong socola 220ml', 'Sua chua uong men song co duong Vinamilk Probi chai 130ml', 'Sua dinh duong co duong Vinamilk A D3 bich 220ml', 'Sua dinh duong khong duong Vinamilk A D3 bich 220ml', 'Sua tiet trung co duong Dutch Lady Canxi Protein 220ml', 'Sua tiet trung khong duong Dutch Lady Canxi Protein 220ml', 'Sua vinamilk loc 4 hop', 'Thit heo ham Vissan hop 150g', 'Tra Sua Huong Dao Cozy 225ml', 'Tra Sua Matcha Cozy 225ml', 'Tuong ot cay nong Cholimex chai 270g', 'Tuong ot chinsu 500g', 'Twister vi cam 445ml', 'Xuc xich LakLak pho mai loc xoay Xixifarm ly 49g', 'Xuc xich Teen tiet trung vi bo LC FOODS goi 175g', 'Xuc xich bo an lien CP Gold 200g', 'Xuc xich bo can lien CP Red 200g', 'Xuc xich bo dinh duong Visan 175g', 'Xuc xich heo hao hang Pinku Oji goi 175g', 'Xuc xich lac xot vi truong muoi 72g', 'Xuc xich pho mai Heo Cao Boi 60g']
        self.metadata =MetadataCatalog.get("dataset_test12")
        #self.metadata = 
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            #out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            #cv2_imshow(out.get_image()[:, :, ::-1])
            #visualized_image = out.get_image()[:, :, ::-1]
            offset = 20
           
            # Converts Matplotlib RGB format to OpenCV BGR format
            cv2.putText(vis_frame.get_image(), "TEST", (5, offset),cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.85, (32, 0, 0), 2)
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            
            #cv2.putText(frame, "{}: {} - {}".format(key, l, l*price), (5, offset),
            #                 cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.85, (32, 0, 0), 2)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
