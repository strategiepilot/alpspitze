import json, os, requests, sys, io
import pandas as pd
import numpy as np

from google.cloud import vision, storage, speech
from google.cloud import videointelligence_v1 as vi
from google.oauth2 import service_account
from pandas import json_normalize
from typing import Optional, Sequence

#================================================================
#                          Variables & Dictionaries
#================================================================



#================================================================
#                          Functions
#================================================================

def connectGoogleAPI():
    creds = "/Users/andreas/Projects/Hollywood/GC Admin/hollywood-313911-c1bfc1227cbe.json"
    credentials = service_account.Credentials.from_service_account_file(creds)
    storage_client = storage.Client(credentials=credentials)
    return credentials, storage_client

def createCollectors():
    colsTVC        = "MEDIAFILE SEGMENT_LABEL SEGM_CATEGORY_LABEL LABEL_CONF TIME_START TIME_END".split()
    colsDetail     = "MEDIAFILE FRAME_TIME FRAME_LABEL FRAME_CATEGORY_LABEL FRAME_CONF".split()
    colsShotDetail = "MEDIAFILE SHOT_LABEL SHOT_CATEGORY_LABEL LABEL_CONF TIME_START TIME_END".split()
    colsLogo       = "MEDIAFILE LOGO ENTITY_ID TIME_START TIME_END LOGO_CONF".split()
    colsShots      = "MEDIAFILE SHOT_ID TIME_START TIME_END".split()
    colsText       = "MEDIAFILE TIME_START TIME_END TEXT TXT_CONF".split()
    colsAudio      = "MEDIAFILE AUDIO AUDIO_CONF WORDS".split()
    dfTVC       = pd.DataFrame(columns=colsTVC)
    dfFrame     = pd.DataFrame(columns=colsDetail)
    dfShotLabel = pd.DataFrame(columns=colsShotDetail)
    dfLogo      = pd.DataFrame(columns=colsLogo)#
    dfShots     = pd.DataFrame(columns=colsShots)
    dfText      = pd.DataFrame(columns=colsText)
    dfAudio     = pd.DataFrame(columns=colsAudio)
    return dfTVC, dfFrame, dfShotLabel, dfLogo, dfShots, dfText, dfAudio

def parseResults_LabelDetection(resultLabel, dfTVC, mediafile, i):
    # Process video/segment level label annotations
    segment_labels = resultLabel.annotation_results[0].segment_label_annotations
    for j, segment_label in enumerate(segment_labels,1):
        # print("Video label description: {}".format(segment_label.entity.description))
        nbr_catEntities = len(segment_label.category_entities)
        # print(len(segment_label.category_entities))

        if nbr_catEntities == 0:
            dfTVC.loc[i*100_000+j*100, ["MEDIAFILE","SEGMENT_LABEL"]] = [mediafile, segment_label.entity.description]
            for l, segment in enumerate(segment_label.segments,1):
                start_time = (segment.segment.start_time_offset.seconds+segment.segment.start_time_offset.microseconds/1e6)
                end_time = (segment.segment.end_time_offset.seconds+segment.segment.end_time_offset.microseconds/1e6)
                positions = "{}s to {}s".format(start_time, end_time)
                confidence = segment.confidence
                # print("\tSegment {}: {}".format(i, positions)); print("\tConfidence: {}".format(confidence))
                dfTVC.loc[i*100_000+j*100, ["TIME_START","TIME_END","LABEL_CONF"]] = [start_time,end_time,confidence]

        else:
            for k, category_entity in enumerate(segment_label.category_entities,1):
                    # print("\tLabel category description: {}".format(category_entity.description))
                    dfTVC.loc[i*100_000+j*100+k, ["MEDIAFILE","SEGMENT_LABEL","SEGM_CATEGORY_LABEL"]] = [mediafile, segment_label.entity.description,category_entity.description]
                    for l, segment in enumerate(segment_label.segments,1):
                        start_time = (segment.segment.start_time_offset.seconds+segment.segment.start_time_offset.microseconds/1e6)
                        end_time = (segment.segment.end_time_offset.seconds+segment.segment.end_time_offset.microseconds/1e6)
                        positions = "{}s to {}s".format(start_time, end_time)
                        confidence = segment.confidence
                        # print("\tSegment {}: {}".format(i, positions)); print("\tConfidence: {}".format(confidence))
                        dfTVC.loc[i*100_000+j*100+k, ["TIME_START","TIME_END","LABEL_CONF"]] = [start_time,end_time,confidence]

def parseResults_FrameLabelDetection(resultLabel, dfFrame, mediafile, i):
    frame_labels = resultLabel.annotation_results[0].frame_label_annotations
    for m, frame_label in enumerate(frame_labels,1):
        nbr_catEntsFrame = len(frame_label.category_entities)
        # print(nbr_catEntsFrame)
        
        if nbr_catEntsFrame == 0:
            # print("Frame label description: {}".format(frame_label.entity.description))
            # Each frame_label_annotation has many frames, here we print information only about the first frame.
            frame = frame_label.frames[0]
            time_offset = frame.time_offset.seconds + frame.time_offset.microseconds / 1e6
            # print("\tFirst frame time offset: {}s".format(time_offset))
            # print("\tFirst frame confidence: {}".format(frame.confidence))
            # print("\n")
            dfFrame.loc[i*100_000+m*100,["MEDIAFILE","FRAME_LABEL","FRAME_TIME", "FRAME_CONF"]] = [mediafile, frame_label.entity.description,time_offset, frame.confidence]

        else:
            for p, category_entity in enumerate(frame_label.category_entities,1):
                # print("\tLabel category description: {}".format(category_entity.description))
                # Each frame_label_annotation has many frames, here we print information only about the first frame.
                frame = frame_label.frames[0]
                time_offset = frame.time_offset.seconds + frame.time_offset.microseconds / 1e6
                # print("\tFirst frame time offset: {}s".format(time_offset))
                # print("\tFirst frame confidence: {}".format(frame.confidence))
                # print("\n")
                dfFrame.loc[i*100_000+m*100+p,["MEDIAFILE","FRAME_LABEL","FRAME_CATEGORY_LABEL","FRAME_TIME", "FRAME_CONF"]] = [mediafile, frame_label.entity.description,category_entity.description,time_offset,frame.confidence]

def parseResults_LogoDetection(resultLogo, dfLogo, mediafile, i):
    # # # Get the first response, since we sent only one video.
    annotation_result = resultLogo.annotation_results[0]
    # Annotations for list of logos detected, tracked and recognized in video.
    for q, logo_recognition_annotation in enumerate(annotation_result.logo_recognition_annotations,1):
        entity = logo_recognition_annotation.entity
        entityID = entity.entity_id
        entityDescription = entity.description

    #     print("-"*100)
    #     print(f"######### NEW IDENTIY - Logo {q}   #########")
    #     print(u"Entity Id : {}".format(entityID))
    #     print(u"Description : {}".format(entityDescription))

    #     # All logo tracks where the recognized logo appears. Each track corresponds
    #     # to one logo instance appearing in consecutive frames.
        for l, track in enumerate(logo_recognition_annotation.tracks,1):
            # Video segment of a track.
            startTime = (track.segment.start_time_offset.seconds + track.segment.start_time_offset.microseconds/1e6)
            endTime = (track.segment.end_time_offset.seconds + track.segment.end_time_offset.microseconds/1e6)
            # print(f"\n\tStart Time Offset: {startTime}")
            # print(f"\tEnd Time Offset: {endTime}")
            # print(u"\tConfidence : {}".format(track.confidence))
            dfLogo.loc[i*100_000+q*1000+l,:] = [mediafile,entityDescription,entityID,startTime,endTime,track.confidence]
  
def parseResults_ShotLabelDetection(resultLabel, dfShotLabel, mediafile, i):
    # Process shot level label annotations
    shot_labels = resultLabel.annotation_results[0].shot_label_annotations
    for y,shot_label in enumerate(shot_labels,1):
        # print("Video label description: {}".format(segment_label.entity.description))
        nbr_catEntities = len(shot_label.category_entities)
        # print(len(segment_label.category_entities))

        if nbr_catEntities == 0:
            dfShotLabel.loc[i*100_000+y*100, ["MEDIAFILE","SHOT_LABEL"]] = [mediafile, shot_label.entity.description]
            for l, shotsegment in enumerate(shot_label.segments,1):
                start_time = (shotsegment.segment.start_time_offset.seconds+shotsegment.segment.start_time_offset.microseconds/1e6)
                end_time = (shotsegment.segment.end_time_offset.seconds+shotsegment.segment.end_time_offset.microseconds/1e6)
                positions = "{}s to {}s".format(start_time, end_time)
                confidence = shotsegment.confidence
                # print("\tSegment {}: {}".format(i, positions)); print("\tConfidence: {}".format(confidence))
                dfShotLabel.loc[i*100_000+y*100, ["TIME_START","TIME_END","LABEL_CONF"]] = [start_time,end_time,confidence]

        else:
            for k, category_entity in enumerate(shot_label.category_entities,1):
                    # print("\tLabel category description: {}".format(category_entity.description))
                    dfShotLabel.loc[i*100_000+y*100+k, ["MEDIAFILE","SHOT_LABEL","SHOT_CATEGORY_LABEL"]] = [mediafile, shot_label.entity.description,category_entity.description]
                    for l, shotsegment in enumerate(shot_label.segments,1):
                        start_time = (shotsegment.segment.start_time_offset.seconds+shotsegment.segment.start_time_offset.microseconds/1e6)
                        end_time = (shotsegment.segment.end_time_offset.seconds+shotsegment.segment.end_time_offset.microseconds/1e6)
                        positions = "{}s to {}s".format(start_time, end_time)
                        confidence = shotsegment.confidence
                        # print("\tSegment {}: {}".format(i, positions)); print("\tConfidence: {}".format(confidence))
                        dfShotLabel.loc[i*100_000+y*100+k, ["TIME_START","TIME_END","LABEL_CONF"]] = [start_time,end_time,confidence]
 
def parseResults_ShotDetection(resultShots, dfShots, mediafile, i):

    # Get the first response, because a single video was processed
    for r, shot in enumerate(resultShots.annotation_results[0].shot_annotations,1):
        start_time = (shot.start_time_offset.seconds + shot.start_time_offset.microseconds/1e6)
        end_time = (shot.end_time_offset.seconds + shot.end_time_offset.microseconds/1e6)
        # print("\tMediafile {} Shot {}: {} to {}".format(i*100_000, r, np.round(start_time,3), np.round(end_time,3)))
        dfShots.loc[i*100_000+r,:] = [mediafile, r, start_time, end_time]

def parseResults_TextDetection(resultText, dfText, mediafile, i):
    # # --------- Fetch results from Text Detection ---------
    # # The first result is retrieved because a single video was processed.
    annotation_result = resultText.annotation_results[0]
    for t, text_annotation in enumerate(annotation_result.text_annotations,1):
        text = text_annotation.text
    # #     print("\nText: {}".format(text))
    # #     # Get the first text segment
        text_segment = text_annotation.segments[0]
        start_time   = text_segment.segment.start_time_offset
        end_time     = text_segment.segment.end_time_offset
        startTime    = start_time.seconds + start_time.microseconds * 1e-6
        endTime      = end_time.seconds + end_time.microseconds * 1e-6
        conf         = text_segment.confidence
    #     print("start_time: {}, end_time: {}".format(startTime, endTime))
    #     print("Confidence: {}".format(text_segment.confidence))
        dfText.loc[i*100_000+t,:] = [mediafile, startTime, endTime, text_annotation.text, conf]
    
def transcribe_speech(
    video_client: str,
    video_uri: str,
    language_code: str,
    segments: Optional[Sequence[vi.VideoSegment]] = None,) -> vi.VideoAnnotationResults:
    # video_client = vi.VideoIntelligenceServiceClient(credentials=credentials)
    features = [vi.Feature.SPEECH_TRANSCRIPTION]
    config = vi.SpeechTranscriptionConfig(
        language_code=language_code,
        enable_automatic_punctuation=True,
    )
    context = vi.VideoContext(
        segments=segments,
        speech_transcription_config=config,
    )
    request = vi.AnnotateVideoRequest(
        input_uri=video_uri,
        features=features,
        video_context=context,
    )
    # print(f'Processing video "{video_uri}"...')
    operation = video_client.annotate_video(request)
    return operation.result().annotation_results[0]  

def parseResults_Speech2Text(
    i: int,
    mediafile: str,
    dfAudio: pd.DataFrame(),
    results: vi.VideoAnnotationResults,
    min_confidence: float = 0.8,):

    def keep_transcription(transcription: vi.SpeechTranscription) -> bool:
        return min_confidence <= transcription.alternatives[0].confidence

    transcriptions = results.speech_transcriptions
    transcriptions = [t for t in transcriptions if keep_transcription(t)]
    # transcript = None

    # print(" Word Timestamps ".center(80, "-"))
    for k, transcription in enumerate(transcriptions,1):
        first_alternative = transcription.alternatives[0]
        confidence = first_alternative.confidence
        transcript = first_alternative.transcript
        wordList = []
        for word in first_alternative.words:
            t1 = word.start_time.total_seconds()
            t2 = word.end_time.total_seconds()
            word = word.word
            #print(f"{confidence:4.0%} | {t1:7.3f} | {t2:7.3f} | {word}")
            wordList.append((word,t1,t2))   # ggf. noch ergänzen um Wortlevel-Confidence

    #print(mediafile,"\n",transcript,"\n",confidence,"\n", wordList)
        dfAudio.loc[i*100+k,:] = [mediafile, transcript, confidence, wordList]
        