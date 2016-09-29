from mpi4py import MPI

func = lambda x: x % 2 == 0
a = [3,1,4,1,5,9,2,6,5,3,5,8,9,7,9,3,2,3,8,4,6,2,6,4,3,3,8,3,2,7,9,5,0]

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

res = []

for i in range(0,len(a),size):
    if rank == 0:
        data = a[i:i+size]
        if len(data) < size:
            data.extend([None] * (size - len(data)))
    else:
        data = []

    data = comm.scatter(data, root=0)
    print("rank", rank, "received", data)

    if data is not None and not func(data):
        data = None

    data = comm.gather(data, root=0)

    comm.Barrier()

    if rank == 0:
        res.extend(x for x in data if x is not None)

if rank == 0:
    print(res)
