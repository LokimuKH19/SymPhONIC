# Hard Constraints Matters for PINNs

In this filefolder, several experiments using various structures of PINN solving Lid-driven cavity flow problem at `Re=100` is conducted. From the final result it's obvious that applying hard constraints in the forward functions of PINN to encode boundary conditions as many as we can would appear to significantly improve the network performance especially in the cases without data supervision.
