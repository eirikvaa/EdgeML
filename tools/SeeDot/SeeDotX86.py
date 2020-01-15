# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import argparse
import datetime
import os
import tempfile

import seedot.common as Common
from seedot.mainX86 import MainX86
import seedot.util as Util


class MainDriverX86:

    def parseArgs(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("-a", "--algo", choices=Common.Algo.All,
                            metavar='', help="Algorithm to run ('bonsai' or 'protonn')")
        parser.add_argument("--train", required=True,
                            metavar='', help="Training set file")
        parser.add_argument("--test", required=True,
                            metavar='', help="Testing set file")
        parser.add_argument("--model", required=True, metavar='',
                            help="Directory containing trained model (output from Bonsai/ProtoNN trainer)")
        # parser.add_argument("-v", "--version", default=Common.Version.Fixed, choices=Common.Version.All, metavar='',
        #                    help="Datatype of the generated code (fixed-point or floating-point)")
        parser.add_argument("--tempdir", metavar='',
                            help="Scratch directory for intermediate files")
        parser.add_argument("-o", "--outdir", metavar='',
                            help="Directory to output the generated X86 files")

        self.args = parser.parse_args()

        # Verify the input files and directory exists
        print("Trying to access", self.args.train, self.args.test, self.args.model)
        assert os.path.isfile(self.args.train), "Training set doesn't exist"
        assert os.path.isfile(self.args.test), "Testing set doesn't exist"
        assert os.path.isdir(self.args.model), "Model directory doesn't exist"

        # Assign or create temporary directory
        if self.args.tempdir is not None:
            assert os.path.isdir(
                self.args.tempdir), "Scratch directory doesn't exist"
            Common.tempdir = self.args.tempdir
        else:
            Common.tempdir = os.path.join(tempfile.gettempdir(
            ), "SeeDot", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            os.makedirs(Common.tempdir, exist_ok=True)

        # Assign or create output directory
        if self.args.outdir is not None:
            assert os.path.isdir(
                self.args.outdir), "Output directory doesn't exist"
            Common.outdir = self.args.outdir
        else:
            Common.outdir = os.path.join(Common.tempdir, "x86_usps_protonn")
            os.makedirs(Common.outdir, exist_ok=True)

    # Not relevant for me since this handles Windows specific things.
    def checkMSBuildPath(self):
        found = False
        for path in Common.msbuildPathOptions:
            if os.path.isfile(path):
                found = True
                Common.msbuildPath = path

        if not found:
            raise Exception(
                "Msbuild.exe not found at the following locations:\n%s\nPlease change the path and run again" % (
                    Common.msbuildPathOptions))

    def run(self):
        # Not relevant for me since this handles Windows specific things.
        if Util.windows():
            self.checkMSBuildPath()

        algo, version, trainingInput, testingInput, modelDir = self.args.algo, Common.Version.Fixed, self.args.train, self.args.test, self.args.model

        print("\n================================")
        print("Executing on %s for X86" % (algo))
        print("--------------------------------")
        print("Train file: %s" % (trainingInput))
        print("Test file: %s" % (testingInput))
        print("Model directory: %s" % (modelDir))
        print("================================\n")

        obj = MainX86(algo, version, Common.Target.X86,
                      trainingInput, testingInput, modelDir, None)
        obj.run()


if __name__ == "__main__":
    obj = MainDriverX86()
    obj.parseArgs()
    obj.run()
