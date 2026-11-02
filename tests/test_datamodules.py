# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import MagicMock, call, patch
from urllib.parse import quote

from rasterio.errors import RasterioIOError

from optimized_cog_streaming.datamodules import (
    construct_paths,
    get_image_sizes,
)


class GetImageSizesTests(unittest.TestCase):
    def test_open_errors_do_not_log_credentials(self):
        sas_token = "?sv=2024-11-04&sp=r&sig=test-secret-signature"
        image_uris, mask_uris = construct_paths(
            "https://example.blob.core.windows.net/data",
            ["missing.tif"],
            is_remote=True,
            sas_token=sas_token,
        )
        for uri in image_uris + mask_uris:
            for detail in (uri, quote(uri, safe=""), "test-secret-signature"):
                with self.subTest(uri=uri, detail=detail):
                    stdout, stderr = io.StringIO(), io.StringIO()
                    error = RasterioIOError("Failed to open: " + detail)
                    with patch(
                        "optimized_cog_streaming.datamodules.rasterio.open",
                        side_effect=error,
                    ) as open_mock, redirect_stdout(stdout), redirect_stderr(
                        stderr
                    ):
                        result = get_image_sizes([uri])

                    open_mock.assert_called_once_with(uri)
                    self.assertEqual(result, ([], None, None, [uri]))
                    self.assertEqual(
                        stdout.getvalue(),
                        "Error opening image at index 0: RasterioIOError\n",
                    )
                    self.assertEqual(stderr.getvalue(), "")

    def test_open_errors_do_not_log_local_paths_or_exception_details(self):
        uri = "/private/data/missing.tif"
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch(
            "optimized_cog_streaming.datamodules.rasterio.open",
            side_effect=OSError("Credential: test-private-token"),
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            result = get_image_sizes([uri])

        self.assertEqual(result, ([], None, None, [uri]))
        self.assertEqual(
            stdout.getvalue(), "Error opening image at index 0: OSError\n"
        )
        self.assertEqual(stderr.getvalue(), "")

    def test_failed_file_does_not_change_successful_metadata(self):
        uris = ["first.tif", "missing.tif", "last.tif"]
        first, last = MagicMock(), MagicMock()
        for src, shape in ((first, (512, 1024)), (last, (1024, 2048))):
            src.__enter__.return_value = src
            src.shape = shape
            src.dtypes = ("uint16",)
            src.profile = {"blockysize": 128, "blockxsize": 256}

        stdout = io.StringIO()
        with patch(
            "optimized_cog_streaming.datamodules.rasterio.open",
            side_effect=[first, RasterioIOError("missing"), last],
        ) as open_mock, redirect_stdout(stdout):
            result = get_image_sizes(uris)

        self.assertEqual(open_mock.call_args_list, [call(uri) for uri in uris])
        self.assertEqual(
            result,
            (
                [(512, 1024), (1024, 2048)],
                (128, 256),
                "uint16",
                ["missing.tif"],
            ),
        )
        self.assertEqual(
            stdout.getvalue(),
            "Error opening image at index 1: RasterioIOError\n",
        )
