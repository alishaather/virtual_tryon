import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

class CaptureScreen extends StatefulWidget {
  final double heightCm;

  const CaptureScreen({super.key, required this.heightCm});

  @override
  State<CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends State<CaptureScreen> {
  CameraController? _controller;
  String? _error;
  XFile? _photo;
  bool _taking = false;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() => _error = 'No camera found on this device');
        return;
      }

      // front cam, if there isn't one just use whatever is first
      final cam = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );

      final controller = CameraController(
        cam,
        ResolutionPreset.high,
        enableAudio: false,
      );
      await controller.initialize();

      if (!mounted) {
        await controller.dispose();
        return;
      }
      setState(() => _controller = controller);
    } on CameraException catch (e) {
      if (!mounted) return;
      setState(() {
        if (e.code.startsWith('CameraAccessDenied')) {
          _error = 'Camera permission was denied. '
              'Allow it from your phone settings and come back.';
        } else {
          _error = 'Camera error: ${e.description}';
        }
      });
    }
  }

  Future<void> _takePhoto() async {
    final c = _controller;
    if (c == null || !c.value.isInitialized || _taking) return;

    setState(() => _taking = true);
    try {
      final file = await c.takePicture();
      if (!mounted) return;
      setState(() => _photo = file);
    } on CameraException catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Could not take photo: ${e.description}')),
      );
    } finally {
      if (mounted) setState(() => _taking = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Capture')),
      body: _buildBody(),
    );
  }

  Widget _buildBody() {
    if (_error != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Text(_error!, textAlign: TextAlign.center),
        ),
      );
    }

    final c = _controller;
    if (c == null) {
      return const Center(child: CircularProgressIndicator());
    }

    // photo taken, show it with a retake button
    if (_photo != null) {
      return Column(
        children: [
          Expanded(
            child: Center(
              child: Transform.flip(
                flipX: true,
                child: Image.file(File(_photo!.path)),
              ),
            ),
          ),  
          Padding(
            padding: const EdgeInsets.all(16),
            child: FilledButton(
              onPressed: () => setState(() => _photo = null),
              child: const Text('Retake'),
            ),
          ),
        ],
      );
    }

    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.all(8),
          child: Text('Height: ${widget.heightCm} cm'),
        ),
        Expanded(child: Center(child: CameraPreview(c))),
        Padding(
          padding: const EdgeInsets.all(16),
          child: FilledButton.icon(
            onPressed: _taking ? null : _takePhoto,
            icon: const Icon(Icons.camera_alt),
            label: const Text('Capture'),
          ),
        ),
      ],
    );
  }
}