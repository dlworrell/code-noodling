// OSE.c — segmented sieve with 6-wheel; multi-column or JSON output
// Build: cc -O3 -std=c11 -Wall -Wextra -o ose OSE.c -lm
// Usage:
//   ./ose START END                 # auto columns to terminal width
//   ./ose START END --cols 4        # 4 columns
//   ./ose START END --json primes.json
//   ./ose START END --json -        # JSON to stdout
//   ./ose START END --no-list --json primes.json

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <time.h>
#ifdef __unix__
#include <sys/ioctl.h>
#include <unistd.h>
#endif

typedef unsigned char u8;

static void die(const char* m){fprintf(stderr,"ERROR: %s\n",m);exit(2);}

static double now_ms(void){
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
  return ts.tv_sec*1000.0 + ts.tv_nsec/1e6;
}

/* ---------- Simple sieve up to limit (inclusive) ---------- */
static int* simple_sieve(int limit, int* out_n){
  if(limit<1){*out_n=0;return NULL;}
  int n=limit+1;
  u8* isp=(u8*)malloc(n); if(!isp) die("oom simple_sieve");
  memset(isp,1,n); isp[0]=0; if(n>1) isp[1]=0;
  int r=(int)floor(sqrt((double)limit));
  for(int p=2;p<=r;++p) if(isp[p]) for(int q=p*p;q<=limit;q+=p) isp[q]=0;
  int cnt=0; for(int i=2;i<=limit;++i) if(isp[i]) ++cnt;
  int* primes=(int*)malloc(sizeof(int)*cnt); if(!primes) die("oom primes");
  int k=0; for(int i=2;i<=limit;++i) if(isp[i]) primes[k++]=i;
  free(isp); *out_n=cnt; return primes;
}

/* ---------- 6-wheel candidate walk helpers ---------- */
#define BITGET(B,i) (((B)[(i)>>3]>>((i)&7))&1u)
#define BITCLR(B,i) ((B)[(i)>>3] &= (u8)~(1u<<((i)&7)))
#define BITSET(B,i) ((B)[(i)>>3] |= (u8)(1u<<((i)&7)))

typedef struct { long start, end, first; int first_step; } wheel6_map;

static long count_candidates(long start,long end){
  if(end<start) return 0;
  long x=start; int m=(int)(x%6+6)%6;
  while(m!=1 && m!=5){ ++x; m=(m+1)%6; if(x>end) return 0; }
  long cnt=1; int step=(m==1)?4:2;
  for(;;){ x+=step; step=(step==4)?2:4; if(x>end) break; ++cnt; }
  return cnt;
}

static wheel6_map build_wheel6(long start,long end){
  wheel6_map w={start,end,end+1,0};
  if(end<start) return w;
  long x=start; int m=(int)(x%6+6)%6;
  while(m!=1 && m!=5){ ++x; m=(m+1)%6; if(x>end) return w; }
  w.first=x; w.first_step=(m==1)?4:2; return w;
}

static long wheel6_index_of(const wheel6_map* w,long val){
  if(val<w->first || val>w->end) return -1;
  int m=(int)(val%6+6)%6; if(m!=1 && m!=5) return -1;
  long idx=0, x=w->first; int step=w->first_step;
  while(x<val){ x+=step; step=(step==4)?2:4; ++idx; }
  return idx;
}

/* ---------- sieve segment and collect primes ---------- */
typedef struct { long* v; size_t n, cap; } vec_long;

static void vpush(vec_long* a,long x){
  if(a->n==a->cap){ a->cap=a->cap? a->cap*2:256; a->v=realloc(a->v,a->cap*sizeof(long)); if(!a->v) die("oom vec"); }
  a->v[a->n++]=x;
}

static void sieve_collect(long start,long end, vec_long* out){
  int base_n=0; int limit=(int)floor(sqrt((double)end));
  int* base=simple_sieve(limit,&base_n);

  wheel6_map w=build_wheel6(start,end);
  long cand_cnt=count_candidates(start,end);
  long bytes=(cand_cnt+7)>>3; if(bytes==0) bytes=1;
  u8* bits=(u8*)malloc(bytes); if(!bits) die("oom bits");
  memset(bits,0xFF,bytes);

  for(int i=0;i<base_n;++i){
    int p=base[i];
    if(p<5) continue;                 // 2,3 excluded by wheel
    long p2=(long)p*(long)p;
    long q=((start + p - 1)/p)*p;     // ceil(start/p)*p
    if(q<p2) q=p2;
    for(long m=q; m<=end; m+=p){
      int mod=(int)(m%6+6)%6; if(mod!=1 && mod!=5) continue;
      long idx=wheel6_index_of(&w,m); if(idx>=0) BITCLR(bits,idx);
    }
  }

  if(start<=2 && 2<=end) vpush(out,2);
  if(start<=3 && 3<=end) vpush(out,3);

  long x=w.first; if(x<=end){
    int step=w.first_step; long idx=0;
    while(x<=end){
      if(BITGET(bits,idx) && x>=5) vpush(out,x);
      x+=step; step=(step==4)?2:4; ++idx;
    }
  }
  free(bits); free(base);
}

/* ---------- output helpers ---------- */
static int term_width(void){
#ifdef __unix__
  struct winsize w;
  if(isatty(STDOUT_FILENO) && ioctl(STDOUT_FILENO,TIOCGWINSZ,&w)==0 && w.ws_col>0)
    return (int)w.ws_col;
#endif
  return 100; // sensible default
}

static void print_columns(const long* a,size_t n,int cols){
  if(n==0){ puts(""); return; }
  // find max width
  long mx=a[n-1]; int w=1; while(mx){ ++w; mx/=10; }
  int colw=w+1;                      // 1 space gap
  if(cols<=0){                       // auto-fit by terminal width
    int tw=term_width();
    cols = (tw>colw)? (tw/colw) : 1;
  }
  size_t rows=(n+cols-1)/cols;
  for(size_t r=0;r<rows;++r){
    for(int c=0;c<cols;++c){
      size_t i=c*rows + r;
      if(i<n) printf("%*ld", w, a[i]);
      if(c<cols-1 && c*rows + r + rows < n) putchar(' ');
    }
    putchar('\n');
  }
}

static int write_json(const char* path,long start,long end,const long* a,size_t n){
  FILE* f = (path && strcmp(path,"-")==0) ? stdout : fopen(path,"w");
  if(!f) return -1;
  fprintf(f,"{\n");
  fprintf(f,"  \"range\": {\"start\": %ld, \"end\": %ld},\n", start, end);
  fprintf(f,"  \"count\": %zu,\n", n);
  fprintf(f,"  \"primes\": [");
  for(size_t i=0;i<n;++i){
    if(i) fputc(',',f);
    if((i%16)==0){ fputc('\n',f); fputs("    ",f); }
    fprintf(f,"%ld", a[i]);
  }
  if(n) fputc('\n',f);
  fprintf(f,"  ]\n}\n");
  if(f!=stdout) fclose(f);
  return 0;
}

/* ---------- main ---------- */
int main(int argc,char** argv){
  if(argc<3){
    fprintf(stderr,"Usage: %s START END [--cols N] [--json FILE|-] [--no-list]\n", argv[0]);
    return 2;
  }
  long start = atol(argv[1]);
  long end   = atol(argv[2]);
  if(end<start) die("END < START");

  int forced_cols = 0;
  const char* json_path = NULL;
  int no_list = 0;

  for(int i=3;i<argc;++i){
    if(strcmp(argv[i],"--cols")==0 && i+1<argc){ forced_cols = atoi(argv[++i]); continue; }
    if(strcmp(argv[i],"--json")==0 && i+1<argc){ json_path = argv[++i]; continue; }
    if(strcmp(argv[i],"--no-list")==0){ no_list = 1; continue; }
    fprintf(stderr,"Unknown option: %s\n", argv[i]); return 2;
  }

  double t0=now_ms();
  vec_long primes={0,0,0};
  sieve_collect(start,end,&primes);
  double t1=now_ms();

  if(json_path){
    if(write_json(json_path,start,end,primes.v,primes.n)!=0)
      fprintf(stderr,"WARN: could not write JSON to %s\n", json_path);
  }

  if(!no_list){
    print_columns(primes.v, primes.n, forced_cols);
  }

  fprintf(stderr,"[ose] range=[%ld,%ld] count=%zu time=%.2f ms\n",
          start,end,primes.n,(t1-t0));

  free(primes.v);
  return 0;
}