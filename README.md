# camfi
**C**amera-based **A**nalysis and **M**onitoring of **F**lying **I**nsects

# Image annotation

## Performing the annotations in VIA

1. Open “via.html”.

2. In the top menu, click on Project, then choose Load (see the red oval to the
   left in below screenshot). Find your VIA project file, and click Open.

   ![Loading and navigating VIA](manual_figures/navigating_via.png)

3. If it is the first time that you work on the file, simply start with the
   first image. If you have already worked on the project file before and you
   have a saved version, scroll down to the last image that you were working on
   and click on it. You can now start working from that image.

4. You move between images (backwards and forwards) with the sideways arrows in
   the top menu (see the blue oval to the right in Fig.1), or you can use the
   sideways arrows on your keyboard.

5. To zoom in and out, use the magnifying glass (+ or -, see the yellow oval in
   the upper right corner in Fig.1).

6. To the left, you can find different Region shapes (see the red oval in
   Fig.2). The only ones I have been using are the “Circular region shape”, the
   “Point region shape”, and the “Polyline region shape”.

   ![Region shapes in VIA](manual_figures/region_shapes.png)

   - Circular region shape: This shape can be used when you cannot see the
     whole moth (or the whole motion blur), e.g., when the moth is going out the
     edge of the image (see the moth in the upper right corner in Fig.3), if
     another moth or object is covering it, or if you find it hard to see where
     the motion blur starts and ends. To draw a circle region, simply press single
     click and drag the mouse.
   - Point region shape: This shape can be used when the moth is visible as a
     point (usually in brighter conditions; see the two moths in Fig.4). There
     is not as much motion blur, because the sun has not set yet, meaning the
     camera used a shorter exposure time. It can also be used when the area of
     the moth is too small for the circular region shape to function. When this
     is the case, an error message will show up at the bottom of the screen. To
     define a point, press single click.
   - Polyline region shape: This shape should be used when the moth is
     visible as a line (due to motion blur). Often, you can see the flapping of the
     wings (see Fig.3). To draw a polyline, single click on the start of the motion
     blur, and then at the end of the motion blur. To finish drawing the polyline,
     press “Enter” on the keyboard. It is important to make sure that the ends of
     the polyline annotations match up with the ends of the motion blur. Also
     important is to follow the line carefully - by clicking along the line several
     times - so that a bend is properly annotated (see the polyline in Fig.5).
