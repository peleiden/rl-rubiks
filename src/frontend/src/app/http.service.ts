import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { ICubeResponse, IActionRequest, IScrambleRequest, IScrambleResponse, IInfoResponse, ISolveRequest, ISolveResponse } from './rubiks/rubiks';

const urls = {
  info: "http://127.0.0.1:5000/info",
  solved: "http://127.0.0.1:5000/solved",
  action: "http://127.0.0.1:5000/action",
  scramble: "http://127.0.0.1:5000/scramble",
  solve: "http://127.0.0.1:5000/solve",
}

@Injectable({
  providedIn: 'root'
})
export class HttpService {

  constructor(private http: HttpClient) { }

  public async getInfo() {
    return this.http.get<IInfoResponse>(urls.info).toPromise();
  }

  public async getSolved() {
    return this.http.get<ICubeResponse>(urls.solved).toPromise();
  }

  public async performAction(actionRequest: IActionRequest) {
    return this.http.post<ICubeResponse>(urls.action, actionRequest).toPromise();
  }

  public async scramble(scrambleRequest: IScrambleRequest) {
    return this.http.post<IScrambleResponse>(urls.scramble, scrambleRequest).toPromise();
  }

  public async solve(solveRequest: ISolveRequest) {
    return this.http.post<ISolveResponse>(urls.solve, solveRequest).toPromise();
  }
}
