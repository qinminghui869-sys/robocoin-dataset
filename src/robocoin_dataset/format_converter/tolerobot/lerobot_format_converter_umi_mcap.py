"""LeRobot format converter for UMI data captured by GenRobot DAS Gripper.

The UMI samples are MCAP files that store Foxglove protobuf messages rather
than ROS CDR messages. Images are H.264 payloads in foxglove.CompressedImage;
the gripper opening is stored in foxglove.MagneticEncoderMeasurement.value.
"""

import logging
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from mcap.records import Channel, Message, Schema
from mcap.stream_reader import StreamReader

from robocoin_dataset.format_converter.tolerobot.constant import (
    ACTION_KEY,
    ARGS_KEY,
    CAM_NAME_KEY,
    FEATURES_KEY,
    IMAGE_KEY,
    OBSERVATION_KEY,
    STATE_KEY,
    SUB_ACTION_KEY,
    SUB_STATE_KEY,
)
from robocoin_dataset.format_converter.tolerobot.lerobot_format_converter import (
    LerobotFormatConverter,
)

logger = logging.getLogger(__name__)


@dataclass
class UmiEpisodeCache:
    task_path: Path | None = None
    ep_idx: int | None = None
    is_test: bool = False
    data: dict[str, list[np.ndarray]] | None = None


class FoxgloveProtobufDecoder:
    """Dynamic protobuf decoder backed by MCAP FileDescriptorSet schemas."""

    def __init__(self) -> None:
        self._classes: dict[int, type] = {}

    def decode(self, schema: Schema, data: bytes) -> Any:
        msg_cls = self._classes.get(schema.id)
        if msg_cls is None:
            msg_cls = self._build_message_class(schema)
            self._classes[schema.id] = msg_cls
        msg = msg_cls()
        msg.ParseFromString(data)
        return msg

    @staticmethod
    def _build_message_class(schema: Schema) -> type:
        fds = descriptor_pb2.FileDescriptorSet()
        fds.ParseFromString(schema.data)

        pool = descriptor_pool.DescriptorPool()
        for file_descriptor in fds.file:
            try:
                pool.Add(file_descriptor)
            except Exception:
                # Duplicate dependencies can appear across MCAP schemas.
                pass

        descriptor = pool.FindMessageTypeByName(schema.name)
        return message_factory.GetMessageClass(descriptor)


class LerobotFormatConverterUmiMcap(LerobotFormatConverter):
    """Converter for a directory of UMI MCAP episodes."""

    def __init__(
        self,
        dataset_path: str,
        output_path: str,
        converter_config: dict,
        repo_id: str,
        device_model: str | None = None,
        logger: logging.Logger | None = None,
        video_backend: str = "pyav",
        image_writer_processes: int = 4,
        image_writer_threads: int = 4,
        **kwargs: Any,
    ) -> None:
        self._cache = UmiEpisodeCache()
        self._protobuf_decoder = FoxgloveProtobufDecoder()
        self._default_task = converter_config.get(
            "task_description",
            "operate object with GenRobot DAS Gripper",
        )
        self._mcap_files_by_task: dict[Path, list[Path]] = {}

        super().__init__(
            dataset_path=dataset_path,
            output_path=output_path,
            converter_config=converter_config,
            repo_id=repo_id,
            device_model=device_model,
            logger=logger or logging.getLogger(__name__),
            video_backend=video_backend,
            image_writer_processes=image_writer_processes,
            image_writer_threads=image_writer_threads,
            **kwargs,
        )

    def _get_tasks(self) -> list[str]:
        return [self._default_task]

    def _get_dataset_task_paths(self) -> dict[Path, str]:
        return {self.dataset_path: self._default_task}

    def _prevalidate_files(self) -> None:
        for task_path in self.path_task_dict:
            mcap_files = sorted(task_path.glob("*.mcap"))
            if not mcap_files:
                raise FileNotFoundError(f"No .mcap files found in UMI dataset path: {task_path}")

            self._mcap_files_by_task[task_path] = mcap_files
            self._validate_first_mcap_channels(mcap_files[0])

    def _validate_first_mcap_channels(self, mcap_file: Path) -> None:
        required_topics = self._configured_topics()
        available_topics = self._read_channel_topics(mcap_file)
        missing_topics = sorted(required_topics - available_topics)
        if missing_topics:
            raise ValueError(
                "UMI MCAP is missing required topics:\n"
                + "\n".join(f"  - {topic}" for topic in missing_topics)
            )

    @staticmethod
    def _read_channel_topics(mcap_file: Path) -> set[str]:
        topics: set[str] = set()
        with mcap_file.open("rb") as stream:
            for record in StreamReader(stream).records:
                if isinstance(record, Channel):
                    topics.add(record.topic)
        return topics

    def _configured_topics(self) -> set[str]:
        topics: set[str] = set()
        for image_config in self.converter_config[FEATURES_KEY][OBSERVATION_KEY][IMAGE_KEY]:
            topics.add(image_config[ARGS_KEY]["mcap_topic"])
        for state_config in self.converter_config[FEATURES_KEY][OBSERVATION_KEY][STATE_KEY][
            SUB_STATE_KEY
        ]:
            topics.add(state_config[ARGS_KEY]["mcap_topic"])
        for action_config in self.converter_config[FEATURES_KEY][ACTION_KEY][SUB_ACTION_KEY]:
            topics.add(action_config[ARGS_KEY]["mcap_topic"])
        return topics

    def _get_mcap_file(self, task_path: Path, ep_idx: int) -> Path:
        mcap_files = self._mcap_files_by_task.get(task_path)
        if mcap_files is None:
            mcap_files = sorted(task_path.glob("*.mcap"))
            self._mcap_files_by_task[task_path] = mcap_files

        if ep_idx >= len(mcap_files):
            raise IndexError(
                f"UMI episode index {ep_idx} out of range; found {len(mcap_files)} MCAP files."
            )
        return mcap_files[ep_idx]

    def _get_task_episodes_num(self, task_path: Path) -> int:
        return len(self._mcap_files_by_task.get(task_path, sorted(task_path.glob("*.mcap"))))

    def _get_episode_frames_num(self, task_path: Path, ep_idx: int) -> int:
        if (
            self._cache.task_path == task_path
            and self._cache.ep_idx == ep_idx
            and self._cache.data is not None
        ):
            return len(self._cache.data[self._primary_image_topic()])
        return len(self._get_episode_data(task_path, ep_idx)[self._primary_image_topic()])

    def _prepare_episode_images_buffer(
        self, task_path: Path, ep_idx: int, is_test: bool = False
    ) -> dict[str, list[np.ndarray]]:
        return self._get_episode_data(task_path, ep_idx, is_test=is_test)

    def _prepare_episode_states_buffer(
        self, task_path: Path, ep_idx: int, is_test: bool = False
    ) -> dict[str, list[np.ndarray]]:
        return self._get_episode_data(task_path, ep_idx, is_test=is_test)

    def _prepare_episode_actions_buffer(
        self, task_path: Path, ep_idx: int, is_test: bool = False
    ) -> dict[str, list[np.ndarray]]:
        return self._get_episode_data(task_path, ep_idx, is_test=is_test)

    def _get_frame_image(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        args_dict: dict,
        images_buffer: Any = None,
        sample_only: bool = False,
    ) -> np.ndarray:
        buffer = images_buffer or self._get_episode_data(task_path, ep_idx, is_test=sample_only)
        return buffer[args_dict["mcap_topic"]][frame_idx]

    def _get_frame_sub_states(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        args_dict: dict,
        sub_states_buffer: Any = None,
    ) -> np.ndarray:
        buffer = sub_states_buffer or self._get_episode_data(task_path, ep_idx)
        return self._slice_feature(buffer[args_dict["mcap_topic"]][frame_idx], args_dict)

    def _get_frame_sub_actions(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        args_dict: dict,
        sub_actions_buffer: Any = None,
    ) -> np.ndarray:
        buffer = sub_actions_buffer or self._get_episode_data(task_path, ep_idx)
        return self._slice_feature(buffer[args_dict["mcap_topic"]][frame_idx], args_dict)

    @staticmethod
    def _slice_feature(value: np.ndarray, args_dict: dict) -> np.ndarray:
        range_from = args_dict.get("range_from", 0)
        range_to = args_dict.get("range_to")
        if range_to is None:
            return value[range_from:].astype(np.float32)
        return value[range_from:range_to].astype(np.float32)

    def _get_episode_data(
        self, task_path: Path, ep_idx: int, is_test: bool = False
    ) -> dict[str, list[np.ndarray]]:
        if (
            self._cache.task_path == task_path
            and self._cache.ep_idx == ep_idx
            and self._cache.is_test == is_test
            and self._cache.data is not None
        ):
            return self._cache.data

        mcap_file = self._get_mcap_file(task_path, ep_idx)
        raw_data = self._read_mcap(mcap_file, is_test=is_test)
        aligned_data = self._align_to_primary_camera(raw_data)

        self._cache = UmiEpisodeCache(
            task_path=task_path,
            ep_idx=ep_idx,
            is_test=is_test,
            data=aligned_data,
        )
        return aligned_data

    def _read_mcap(self, mcap_file: Path, is_test: bool = False) -> dict[str, list[tuple[int, np.ndarray]]]:
        image_topics = self._configured_image_topics()
        scalar_topics = self._configured_scalar_topics()
        requested_topics = image_topics | scalar_topics

        topic_data: dict[str, list[tuple[int, np.ndarray]]] = {
            topic: [] for topic in requested_topics
        }
        channels: dict[int, Channel] = {}
        schemas: dict[int, Schema] = {}
        h264_decoders: dict[str, av.CodecContext] = {}
        max_image_frames = 12 if is_test else None

        with mcap_file.open("rb") as stream:
            for record in StreamReader(stream).records:
                if isinstance(record, Schema):
                    schemas[record.id] = record
                    continue
                if isinstance(record, Channel):
                    channels[record.id] = record
                    continue
                if not isinstance(record, Message):
                    continue

                channel = channels[record.channel_id]
                topic = channel.topic
                if topic not in requested_topics:
                    continue

                schema = schemas[channel.schema_id]
                if topic in image_topics:
                    if max_image_frames is not None and len(topic_data[topic]) >= max_image_frames:
                        continue
                    msg = self._protobuf_decoder.decode(schema, record.data)
                    decoder = h264_decoders.setdefault(topic, av.CodecContext.create("h264", "r"))
                    for frame in self._decode_h264_payload(decoder, msg.data):
                        topic_data[topic].append(
                            (record.log_time, frame.to_ndarray(format="rgb24"))
                        )
                elif topic in scalar_topics:
                    msg = self._protobuf_decoder.decode(schema, record.data)
                    topic_data[topic].append(
                        (record.log_time, np.array([msg.value], dtype=np.float32))
                    )

        return topic_data

    @staticmethod
    def _decode_h264_payload(decoder: av.CodecContext, data: bytes) -> list[av.VideoFrame]:
        frames: list[av.VideoFrame] = []
        try:
            packets = decoder.parse(data)
        except Exception:
            return frames

        for packet in packets:
            try:
                frames.extend(decoder.decode(packet))
            except Exception:
                continue
        return frames

    def _align_to_primary_camera(
        self, raw_data: dict[str, list[tuple[int, np.ndarray]]]
    ) -> dict[str, list[np.ndarray]]:
        primary_topic = self._primary_image_topic()
        primary_messages = raw_data.get(primary_topic, [])
        if not primary_messages:
            raise ValueError(f"No decodable primary camera frames found: {primary_topic}")

        max_frames = min(len(messages) for messages in raw_data.values() if messages)
        primary_timestamps = [timestamp for timestamp, _ in primary_messages[:max_frames]]

        aligned_data: dict[str, list[np.ndarray]] = {}
        for topic, messages in raw_data.items():
            if not messages:
                raise ValueError(f"No messages found for configured UMI topic: {topic}")

            timestamps = [timestamp for timestamp, _ in messages]
            values = [value for _, value in messages]
            aligned_data[topic] = [
                values[self._nearest_index(timestamps, target_timestamp)]
                for target_timestamp in primary_timestamps
            ]

        if self.logger:
            summary = ", ".join(
                f"{topic}: {len(values)}" for topic, values in sorted(aligned_data.items())
            )
            self.logger.info(f"Aligned UMI episode to {max_frames} frames ({summary})")

        return aligned_data

    @staticmethod
    def _nearest_index(timestamps: list[int], target_timestamp: int) -> int:
        insert_at = bisect_left(timestamps, target_timestamp)
        if insert_at <= 0:
            return 0
        if insert_at >= len(timestamps):
            return len(timestamps) - 1

        before = insert_at - 1
        after = insert_at
        if target_timestamp - timestamps[before] <= timestamps[after] - target_timestamp:
            return before
        return after

    def _primary_image_topic(self) -> str:
        image_config = self.converter_config[FEATURES_KEY][OBSERVATION_KEY][IMAGE_KEY][0]
        return image_config[ARGS_KEY]["mcap_topic"]

    def _configured_image_topics(self) -> set[str]:
        return {
            image_config[ARGS_KEY]["mcap_topic"]
            for image_config in self.converter_config[FEATURES_KEY][OBSERVATION_KEY][IMAGE_KEY]
        }

    def _configured_scalar_topics(self) -> set[str]:
        topics: set[str] = set()
        for state_config in self.converter_config[FEATURES_KEY][OBSERVATION_KEY][STATE_KEY][
            SUB_STATE_KEY
        ]:
            topics.add(state_config[ARGS_KEY]["mcap_topic"])
        for action_config in self.converter_config[FEATURES_KEY][ACTION_KEY][SUB_ACTION_KEY]:
            topics.add(action_config[ARGS_KEY]["mcap_topic"])
        return topics

    def _get_episode_source_files(self, task_path: Path, ep_idx: int) -> dict:
        mcap_file = self._get_mcap_file(task_path, ep_idx)
        return {
            "format": "MCAP/Foxglove protobuf",
            "mcap_file": str(mcap_file.relative_to(self.dataset_path)),
            "absolute_path": str(mcap_file.absolute()),
        }

    def _cleanup_episode_resources(self) -> None:
        self._cache = UmiEpisodeCache()
