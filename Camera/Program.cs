using System;
using System.Drawing;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using Emgu.CV.VideoSurveillance;

namespace Camera
{
	class Program
	{
		static Timer _myTimer;
		static Capture _capture;
		static ImageViewer _viewer;
		static readonly BackgroundSubtractorMOG2 MogDetector = new BackgroundSubtractorMOG2(0, 32.0f, true);
		static int _frameNumber;

		private static Image<Gray, byte> _previousFrame;
		private static PointF[] _previousFeatures;
		
		static void Main(string[] args)
		{
			const int fps = 30;

			_myTimer = new Timer();
			_myTimer.Interval = 1000 / fps;
			_myTimer.Tick += MyTimerTick;
			_myTimer.Start();

			_capture = new Capture(@"C:\source\dungeon\ComputerVision\Camera\App_Data\converted.avi");
			_viewer = new ImageViewer();
			_viewer.ShowDialog(); 
		}

		private static void MyTimerTick(object sender, EventArgs e)
		{
			var rawFrame = _capture.QueryFrame();
			var originalRoi = rawFrame.ROI;
			var roi = new Rectangle(0, rawFrame.Height / 2, rawFrame.Width, rawFrame.Height - rawFrame.Height / 2);
			
			var cropFrame = rawFrame
				.GetSubRect(roi)
				.Convert<Gray, byte>()
				.SmoothGaussian(17);

			if (_frameNumber++ < 100) 
				MogDetector.Update(cropFrame.Convert<Bgr, byte>());
			else
				MogDetector.Update(cropFrame.Convert<Bgr, byte>(), 1.0e-5);
			
			//Find the motion
			var motionMask = MogDetector.ForegroundMask
				.ThresholdBinary(new Gray(250), new Gray(255)) // Removes shadows
				//.Erode(3).Dilate(3) // Removes small artifacts that aren't really moving
				;

			//var motion = frame.Copy(motionMask);
			//var motion = rawFrame.Add(motionMask);
			var motion = rawFrame.Copy();
			motion.ROI = roi;
			motion.Add(motionMask.Convert<Bgr, byte>());

			if (_previousFrame != null)
			{
				PointF[] returnFeatures;
				byte[] status;
				float[] trackError;
				
				_previousFeatures = _previousFrame.GoodFeaturesToTrack(100, .01, .1, 3)[0];
				OpticalFlow.PyrLK(_previousFrame, motionMask, _previousFeatures, new Size(15, 15), 5, new MCvTermCriteria(5), out returnFeatures, out status, out trackError);

				//Console.WriteLine("Features Found:" + returnFeatures.Length);
				for (var i = 0; i < returnFeatures.Length; i++)
				{
					var prevPoint = _previousFeatures[i];
					var currPoint = returnFeatures[i];
					var state = status[i];
					var error = trackError[i];

					var line = new LineSegment2D(new Point((int)prevPoint.X, (int)prevPoint.Y), new Point((int)currPoint.X, (int)currPoint.Y));
					var travelUp = prevPoint.Y - currPoint.Y;
					var travelLeft = prevPoint.X - currPoint.X;

					//if (state != 1 || !(line.Length > 15) || !(error < 10)) 
					if (!(state == 1 && line.Length > 15 && error < 10))
						continue;

					motion.Draw(line, new Bgr(Color.Red), 3);
					Console.WriteLine("Error: {0}, Up: {1}, Left: {2}, Length: {3}", 
						error, travelUp, travelLeft, line.Length);
				}
			}
			_previousFrame = motionMask;

			motion.ROI = originalRoi;
			_viewer.Image = motion;
		}
	}
}
