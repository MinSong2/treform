import treform as ptm

test='이건진짜좋은영화라라랜드진짜좋은영화'

model_path='../model/korean_segmentation_model.crfsuite'
segmentation=ptm.segmentation.SegmentationKorean(model_path)
correct=segmentation(test)
print(correct)

lstm_model_path='../treform/segmentation/model'
lstm_segmentation=ptm.segmentation.LSTMSegmentationKorean(lstm_model_path)
lstm_correct=lstm_segmentation(test)
print(lstm_correct)

lstm_segmentation.close()