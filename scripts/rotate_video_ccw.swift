import AVFoundation
import CoreGraphics
import Foundation

enum RotateError: Error {
  case invalidArguments
  case noVideoTrack
  case cannotCreateTrack
  case cannotCreateExportSession
  case exportFailed(String)
}

func rotateCCW90(input: URL, output: URL) throws {
  let asset = AVURLAsset(url: input)
  guard let sourceVideoTrack = asset.tracks(withMediaType: .video).first else {
    throw RotateError.noVideoTrack
  }

  let composition = AVMutableComposition()
  guard let videoTrack = composition.addMutableTrack(
    withMediaType: .video,
    preferredTrackID: kCMPersistentTrackID_Invalid
  ) else {
    throw RotateError.cannotCreateTrack
  }

  try videoTrack.insertTimeRange(
    CMTimeRange(start: .zero, duration: asset.duration),
    of: sourceVideoTrack,
    at: .zero
  )

  if let sourceAudioTrack = asset.tracks(withMediaType: .audio).first,
     let audioTrack = composition.addMutableTrack(
       withMediaType: .audio,
       preferredTrackID: kCMPersistentTrackID_Invalid
     ) {
    try audioTrack.insertTimeRange(
      CMTimeRange(start: .zero, duration: asset.duration),
      of: sourceAudioTrack,
      at: .zero
    )
  }

  let sourceSize = sourceVideoTrack.naturalSize
  let renderSize = CGSize(width: sourceSize.height, height: sourceSize.width)

  let instruction = AVMutableVideoCompositionInstruction()
  instruction.timeRange = CMTimeRange(start: .zero, duration: asset.duration)

  let layerInstruction = AVMutableVideoCompositionLayerInstruction(assetTrack: videoTrack)

  // Counterclockwise 90 degrees.
  let ccw = CGAffineTransform(translationX: 0, y: sourceSize.width).rotated(by: -.pi / 2)
  let transform = sourceVideoTrack.preferredTransform.concatenating(ccw)
  layerInstruction.setTransform(transform, at: .zero)
  instruction.layerInstructions = [layerInstruction]

  let videoComposition = AVMutableVideoComposition()
  videoComposition.instructions = [instruction]
  videoComposition.renderSize = renderSize
  videoComposition.frameDuration = CMTime(value: 1, timescale: 30)

  if FileManager.default.fileExists(atPath: output.path) {
    try FileManager.default.removeItem(at: output)
  }

  guard let exporter = AVAssetExportSession(
    asset: composition,
    presetName: AVAssetExportPresetHighestQuality
  ) else {
    throw RotateError.cannotCreateExportSession
  }

  exporter.outputURL = output
  exporter.outputFileType = .mp4
  exporter.shouldOptimizeForNetworkUse = true
  exporter.videoComposition = videoComposition

  let semaphore = DispatchSemaphore(value: 0)
  exporter.exportAsynchronously {
    semaphore.signal()
  }
  semaphore.wait()

  if exporter.status != .completed {
    throw RotateError.exportFailed(exporter.error?.localizedDescription ?? "unknown error")
  }
}

guard CommandLine.arguments.count == 3 else {
  fputs("Usage: swift rotate_video_ccw.swift <input.mp4> <output.mp4>\n", stderr)
  throw RotateError.invalidArguments
}

let input = URL(fileURLWithPath: CommandLine.arguments[1])
let output = URL(fileURLWithPath: CommandLine.arguments[2])
try rotateCCW90(input: input, output: output)
print("Rotated: \(input.lastPathComponent) -> \(output.lastPathComponent)")
