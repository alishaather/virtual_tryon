import 'package:flutter/material.dart';

class CaptureScreen extends StatelessWidget {
  final double heightCm;

  const CaptureScreen({super.key, required this.heightCm});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Capture')),
      body: Center(
        child: Text('Height: $heightCm cm\ncamera comes here'),
      ),
    );
  }
}