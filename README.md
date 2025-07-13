This is a very basic WebGPU-based black hole visualization. It uses Dormand-Prince RK45 to integrate light rays in a Kerr spacetime in real time using your GPU. The project is just a static page. You can run a simple server using the following command in the project root:

``` sh
python -m http.server
```

The performance is not great. On my laptop I get roughly 1~2 seconds per frame (spf, not fps). I'd like to know ways to optimize this to achieve real-time interactivity.
